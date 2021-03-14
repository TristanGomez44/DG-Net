case $1 in
  "nbParts_freeze")
    python3 traintest.py --config outputs/E0.5new_reid0.5_w30000/config.yaml --name E0.5new_reid0.5_w30000 \
                         --model_id widerRange_moreHypParam_nbParts_freeze --freeze_gen_dis True \
                         --gpu_ids 0,1 --max_batch_size 24
    ;;
  "nbParts_freeze_merge")
    python3 traintest.py --config outputs/E0.5new_reid0.5_w30000/config.yaml --name E0.5new_reid0.5_w30000 \
                         --model_id widerRange_moreHypParam_nbParts_freeze_merge --freeze_gen_dis True \
                         --gpu_ids 0,1 --max_batch_size 24 --mergevec True
    ;;
  "bcnn")
    python3 traintest.py --config outputs/E0.5new_reid0.5_w30000/config.yaml --name E0.5new_reid0.5_w30000 \
                         --model_id widerRange_moreHypParam_nbParts_freeze_bcnn --freeze_gen_dis True \
                         --gpu_ids 0,1 --max_batch_size 24 --b_cnn True
    ;;
  "")
    python3 traintest.py --config outputs/E0.5new_reid0.5_w30000/config.yaml  --name E0.5new_reid0.5_w30000 \
                          --gpu_ids 0,1,2,3 --model_id dist --exp_id market --max_batch_size 24 \
                          --distill widerRange_moreHypParam_nbParts --freeze_gen_dis True
    ;;
  "nopretr")
    python3 traintest.py --config outputs/E0.5new_reid0.5_w30000/config.yaml  --name E0.5new_reid0.5_w30000 \
                          --gpu_ids 1 --model_id nopretr --exp_id market --max_batch_size 8 \
                         --pretr False --epochs 100
    ;;
  "test")
    python3 traintest.py --config outputs/E0.5new_reid0.5_w30000/config.yaml  --name E0.5new_reid0.5_w30000 \
                          --gpu_ids 0,1,2,3 --model_id dist --exp_id market --max_batch_size 28 \
                          --distill widerRange_moreHypParam_nbParts --freeze_gen_dis True --optuna_trial_nb 2
    ;;
  "dila")
    python3 traintest.py --config outputs/E0.5new_reid0.5_w30000/config.yaml  --name E0.5new_reid0.5_w30000 \
                        --gpu_ids 0,1,2,3 --model_id dist_dil --exp_id market --max_batch_size 25 \
                        --distill widerRange_moreHypParam_nbParts --freeze_gen_dis True --dilation True
    ;;
  "*")
    echo "no such model"
    ;;
esac
