python3 processResults.py --ids 153 1083 1256 1131 --inds "0,1,3,4,5,11;0,1,2,3,4,6;2,4,5,8,13,16;2,3,5,9,12,14" --plot_id main  --nrow 12
python3 processResults.py --ids 153 1083 1256 1131 --inds "0;0;2;2" --plot_id main  --nrow 7 \
                          --gradcam_id widerRange_moreHypParam_nbParts_freeze \
                          --main_id widerRange_moreHypParam_nbParts_freeze_bcnn \
                          --high_res_id dist

python3 processResults.py --ids 153 1083 --inds "0,3,5;1,2,3" --plot_id teaser --nrow 6

python3 processResults.py --ids 1083 --inds "0" --plot_id teaser_reid  --nrow 7 \
                          --gradcam_id widerRange_moreHypParam_nbParts_freeze \
                          --main_id widerRange_moreHypParam_nbParts_freeze_bcnn \
                          --high_res_id dist
