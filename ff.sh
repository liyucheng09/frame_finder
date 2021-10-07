#!/bin/bash
source /vol/hpc-software/software/Anaconda3/2020.07/etc/profile.d/conda.sh
conda activate lyc
python /user/HS502/yl02706/open-sesame/frame_finder.py "distilroberta-base", "data/open_sesame_v1_data/fn1.7"