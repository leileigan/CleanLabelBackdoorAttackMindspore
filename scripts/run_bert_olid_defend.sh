CUDA_VISIBLE_DEVICES=7 python defend/bert_sst_defense.py --dataset olid \
--lm_model_path /data/home/ganleilei/bert/gpt2/ \
--clean_data_path data/clean_data/offenseval/ \
--poison_data_path data/clean_data/aux_files/olid/poison100_bert_base_tune_mlm35_cf0.4_ga_top300base_pop30_iter15.pkl \
--clean_model_path /data/home/ganleilei/attack/models/clean_bert_tune_olid_adam_lr2e-5_bs32_weight0.002/epoch0.ckpt \
--backdoor_model_path /data/home/ganleilei/attack/models/bert_base_olid_attack_num50_bert_base_freeze_adam_lr0.003_bs32_weight0.002/ \
--pre_model_path /data/home/ganleilei/bert/bert-base-uncased/ \

--clean_model_mlp_num 0 \
--poison_model_mlp_num 1 \

--clean_model_mlp_dim 768 \
--poison_model_mlp_dim 1024 \

--defense_method SCPN