#!/bin/bash
#SBATCH --job-name=python
#SBATCH --partition=9462-64C-512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -o %J.out            # 脚本执行的输出将被保存在当 %J.out文件下，%j表示作业号;
#SBATCH -e %J.err            # 脚本执行的错误日志将被保存在当 %J.err文件下，%j表示作业号;


source activate reactot_env

#python all_geom_site_embedding.py > all_geom_site_embedding.out

python train_two_stage_geomE_gaussian.py > train_two_stage_geomE_gaussian.out

