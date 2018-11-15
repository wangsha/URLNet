#!/usr/bin/env bash
#emb_modes=(1 2 2 3 3 4 5)
#delimit_modes=(0 0 1 0 1 1 1 )
emb_modes=(1 2 5)
delimit_modes=(0 0 1)
train_size=10000
test_size=10000
nb_epoch=5
data_dir='./data'

add_expert_feature=1

task(){
    python train_cnn_expert_feature.py --data.data_dir ${data_dir}/train_${train_size}.txt \
    --data.dev_pct 0.01 --data.delimit_mode ${delimit_modes[$1]} --data.min_word_freq 1 \
    --train.add_expert_feature ${add_expert_feature} \
    --model.emb_mode ${emb_modes[$1]} --model.emb_dim 32 --model.filter_sizes 3,4,5,6 \
    --train.nb_epochs ${nb_epoch} --train.batch_size 1048 \
    --train.add_expert_feature=${add_expert_feature} \
    --log.print_every 5 --log.eval_every 10 --log.checkpoint_every 10 \
    --log.output_dir runs/${train_size}_emb${emb_modes[$1]}_dlm${delimit_modes[$1]}_32dim_minwf1_1conv3456_${nb_epoch}ep_expert${add_expert_feature}/

    python test.py --data.data_dir ${data_dir}/test_${test_size}.txt \
    --data.delimit_mode ${delimit_modes[$1]} \
    --data.word_dict_dir runs/${train_size}_emb${emb_modes[$1]}_dlm${delimit_modes[$1]}_32dim_minwf1_1conv3456_${nb_epoch}ep_expert${add_expert_feature}/words_dict.p \
    --data.subword_dict_dir runs/${train_size}_emb${emb_modes[$1]}_dlm${delimit_modes[$1]}_32dim_minwf1_1conv3456_${nb_epoch}ep_expert${add_expert_feature}/subwords_dict.p \
    --data.char_dict_dir runs/${train_size}_emb${emb_modes[$1]}_dlm${delimit_modes[$1]}_32dim_minwf1_1conv3456_${nb_epoch}ep_expert${add_expert_feature}/chars_dict.p \
    --log.checkpoint_dir runs/${train_size}_emb${emb_modes[$1]}_dlm${delimit_modes[$1]}_32dim_minwf1_1conv3456_${nb_epoch}ep_expert${add_expert_feature}/checkpoints/ \
    --log.output_dir runs/${train_size}_emb${emb_modes[$1]}_dlm${delimit_modes[$1]}_32dim_minwf1_1conv3456_${nb_epoch}ep_expert${add_expert_feature}/train_${train_size}_test_${test_size}.txt \
    --model.emb_mode ${emb_modes[$1]} --model.emb_dim 32 \
    --test.batch_size 1048

    python auc.py --input_path runs/${train_size}_emb${emb_modes[$1]}_dlm${delimit_modes[$1]}_32dim_minwf1_1conv3456_${nb_epoch}ep_expert${add_expert_feature}/ --input_file train_${train_size}_test_${test_size}.txt --threshold 0.5
}

for ((i=0; i <${#emb_modes[@]}; ++i))
    do
        task "$i"
    done
