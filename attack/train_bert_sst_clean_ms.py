import argparse
import os
import sys
import time
sys.path.append("/home/ganleilei/workspace/CleanLabelTextualBackdoorAttackMindspore")

# import numpy as np
# import torch
# import torch.nn as nn
# from models.model import BERT
# from data_preprocess.dataset import BERTDataset, bert_fn
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
#                               TensorDataset)
# from torch.utils.data.distributed import DistributedSampler
# from tqdm import tqdm
# from transformers import AutoTokenizer

import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, set_seed, save_checkpoint
from mindspore.nn import AdamWeightDecay, SGD
from models.model_ms import BERT
from data_preprocess.dataset import Iterable
from mindspore.dataset import SequentialSampler, GeneratorDataset, text
import pandas as pd
from tqdm import tqdm
from mindformers import BertConfig, BertModel, BertTokenizer


SEED=1024
set_seed(SEED)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
ms.context.set_context(device_target="GPU")

def adjust_lr(optimizer):
    lr = optimizer.param_groups[0]['lr']
    for param_group in optimizer.param_groups:
        adjusted_lr = lr * 0.9
        param_group['lr'] = adjusted_lr if adjusted_lr > 1e-5 else 1e-5
        print("Adjusted learning rate: %.4f" % param_group['lr'])

def read_data(file_path):
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


def evaluation(model, loader):
    model.set_train(False)
    total_number = 0
    total_correct = 0

    for datapoint in loader:
        padded_text, attention_masks, labels = datapoint
        output, _ = model(padded_text, attention_masks)
        _, idx = output.argmax(1)
        correct = (idx == labels).asnumpy().sum()
        total_correct += correct
        total_number += labels.size(0)
    acc = total_correct / total_number
    return acc



def train(model: BERT, optimizer, epoch, save_path, train_loader_clean, dev_loader_clean, test_loader_clean):

    best_dev_acc = -1
    last_train_avg_loss = 100000
    loss_fn = nn.CrossEntropyLoss()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    def forward_fn(input_ids, token_type_ids, attention_masks, label):
        output, pooled_output = model(input_ids, token_type_ids, attention_masks)
        loss = loss_fn(output, label)
        return loss, output

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    print(optimizer)
    for idx in range(epoch):
        model.set_train()
        total_loss = 0
        start_time = time.time()
        for datapoint in tqdm(train_loader_clean.create_tuple_iterator()):

            input_ids, token_type_ids, attention_masks, labels = datapoint
            (loss, _), grads = grad_fn(input_ids, token_type_ids, attention_masks, labels)
            optimizer(grads)
            total_loss += loss.asnumpy()
        
        avg_loss = total_loss / len(train_loader_clean)
        dev_clean_acc = evaluation(model, dev_loader_clean)
        test_clean_acc = evaluation(model, test_loader_clean)

        print('Epoch %d finish training, cost: %.2fs, avg loss: %.4f, dev clean acc: %.4f, test clean acc: %.4f' % (
            idx, time.time() - start_time, avg_loss, dev_clean_acc, test_clean_acc))

        if dev_clean_acc > best_dev_acc:
            best_dev_acc = dev_clean_acc 
            save_checkpoint(model, os.path.join(save_path, f"epoch{idx}.ckpt"))
        
        if avg_loss > last_train_avg_loss:
            print('Lloss rise, need to adjust lr, current lr: {}'.format(optimizer.param_groups[0]['lr']))
            adjust_lr(optimizer)

        last_train_avg_loss = avg_loss
        sys.stdout.flush()



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0)
    parser.add_argument('--data', type=str, default='sst-2')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.002)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--clean_data_path', type=str, default="data/clean_data/sst-2")
    parser.add_argument('--save_path', type=str, default="checkpoint/")
    parser.add_argument('--pre_model_path', default="bert_base_uncased", type=str)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--mlp_layer_num', default=0, type=int)
    parser.add_argument('--mlp_layer_dim', default=768, type=int)

    args = parser.parse_args()
    print(args)

    lr = args.lr
    data_selected = args.data
    batch_size = args.batch_size
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    epoch = args.epoch
    save_path = args.save_path
    mlp_layer_num = args.mlp_layer_num
    mlp_layer_dim = args.mlp_layer_dim
    
    clean_train_data, clean_dev_data, clean_test_data = get_all_data(args.clean_data_path)
    print(f"Training dataset size:{len(clean_train_data)}")
    print(f"Dev dataset size:{len(clean_dev_data)}")
    print(f"Test dataset size:{len(clean_test_data)}")
    clean_train_dataset, clean_dev_dataset, clean_test_dataset = Iterable(
        clean_train_data), Iterable(clean_dev_data), Iterable(clean_test_data)

    clean_train_dataset = GeneratorDataset(clean_train_dataset, column_names=['input_ids', 'token_type_ids', 'attention_mask', 'label'], shuffle=True)
    clean_dev_dataset   = GeneratorDataset(clean_dev_dataset, column_names=['input_ids', 'token_type_ids', 'attention_mask', 'label'], shuffle=True)
    clean_test_dataset  = GeneratorDataset(clean_test_dataset, column_names=['input_ids', 'token_type_ids', 'attention_mask', 'label'], shuffle=False)
    
    print("clean train dataset first sample:", clean_train_dataset.column_names)

    train_loader_clean = clean_train_dataset.batch(batch_size)
    dev_loader_clean = clean_dev_dataset.batch(batch_size)
    test_loader_clean = clean_test_dataset.batch(batch_size)

    class_num = 4 if data_selected=='ag' else 2
    model = BERT(args.pre_model_path, mlp_layer_num, class_num=class_num, hidden_dim=mlp_layer_dim)
    # if args.freeze:
    #     for param in model.bert.parameters():
    #         param.requires_grad = False

    if optimizer == 'adam':
        optimizer = AdamWeightDecay(model.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
    else:
        optimizer = SGD(model.trainable_params(), learning_rate=lr, weight_decay=weight_decay, momentum=0.9)
    
    sys.stdout.flush()
    train(model, optimizer, epoch, save_path, train_loader_clean, dev_loader_clean, test_loader_clean)

if __name__ == '__main__':
    main()
