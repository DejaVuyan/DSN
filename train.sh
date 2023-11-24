# 采用hidden_dim 320来操作
#CUDA_VISIBLE_DEVICES=0,1 python train.py --style_dir ./Image/ --content_dir /data/yuzun/MRI_data/hcp_all/flip/train --save_dir /data/yuzun/StyTR-2_model_320/ --batch_size 6  --hidden_dim 320 --max_iter 10000 --save_model_interval 1000
# 增加了iter的次数，后面的实验大多采用这个结果
#CUDA_VISIBLE_DEVICES=0,1 python train.py --style_dir /data/yuzun/MRI_data/ADNI1_x/A/train --content_dir /data/yuzun/MRI_data/hcp_all/flip/train --save_dir /data/yuzun/StyTR-2_model_320_2/ --batch_size 6  --hidden_dim 320 --max_iter 200000
# 将content data 和style data反过来
#CUDA_VISIBLE_DEVICES=0,1 python train.py --style_dir /home/yuzun/project/MRI_multiModel_SR/StyTR-2/hcp_style --content_dir /data/yuzun/ADNI1_train320 --save_dir /data/yuzun/StyTR-2_model_320_ADNI1/ --batch_size 6  --hidden_dim 320 --max_iter 10000 --save_model_interval 1000
# 探究各种loss对实验结果的影响
#CUDA_VISIBLE_DEVICES=0,1 python train.py --style_dir ./Image/ --content_dir /data/yuzun/MRI_data/hcp_all/flip/train --save_dir /data/yuzun/Stytr_experience/004_StyTR-2_model_content10 --batch_size 6  --hidden_dim 320 --max_iter 200000 --content_weight 10
#CUDA_VISIBLE_DEVICES=0,1 python train.py --style_dir ./Image/ --content_dir /data/yuzun/MRI_data/hcp_all/flip/train --save_dir /data/yuzun/Stytr_experience/005_StyTR-2_model_content15 --batch_size 6  --hidden_dim 320 --max_iter 200000 --content_weight 15
#CUDA_VISIBLE_DEVICES=0,1 python train.py --style_dir ./Image/ --content_dir /data/yuzun/MRI_data/hcp_all/flip/train --save_dir /data/yuzun/Stytr_experience/006_StyTR-2_model_style10 --batch_size 6  --hidden_dim 320 --max_iter 200000 --style_weight 10

# 2022-9-9 使用SE-10的数据集进行训练
#CUDA_VISIBLE_DEVICES=1 python train.py --style_dir ./Image/ --content_dir /data/yuzun/SE_10/SE_10_imgs_flip --save_dir /data/yuzun/Stytr_experience/007_SE10_1 --batch_size 6  --hidden_dim 512 --max_iter 10000 --save_model_interval 1000

# 2022-9-11 008 2号服务器

# 2022-9-12 将007的content loss提高
#CUDA_VISIBLE_DEVICES=0 python train.py --style_dir ./Image/ --content_dir /data/yuzun/SE_10/SE_10_imgs_flip --save_dir /data/yuzun/Stytr_experience/009_SE10_2_c10 --batch_size 3 --hidden_dim 512 --max_iter 50000 --save_model_interval 1000 --content_weight 10
# 2022-9-13 将训练时间延长，希望能够降低loss
# iter改为300000，将ADNI的test,val也加入到style_dir中
#CUDA_VISIBLE_DEVICES=0 python train.py --style_dir /data/yuzun/MRI_data/ADNI1_x/A/all --content_dir /data/yuzun/SE_10/SE_10_imgs_flip --save_dir /data/yuzun/Stytr_experience/010_SE10_2_160000 --batch_size 3 --hidden_dim 512 --max_iter 300000 

# 2022-9-21 使用新来的数据集进行训练
#CUDA_VISIBLE_DEVICES=0 python train.py --style_dir /data/yuzun/MRI_data/ADNI1_x/A/train --content_dir /data/yuzun/SE_0919/SE_0919_HR_flip --save_dir /data/yuzun/Stytr_experience/011_SE919_1 --batch_size 3  --hidden_dim 512 --max_iter 10000 --save_model_interval 1000

# 2022-9-21 将上一个实验的epoch增大，上一个模糊得太多了
#CUDA_VISIBLE_DEVICES=0 python train.py --style_dir /data/yuzun/MRI_data/ADNI1_x/A/train --content_dir /data/yuzun/SE_0919/SE_0919_HR_flip --save_dir /data/yuzun/Stytr_experience/011_SE919_1 --batch_size 3  --hidden_dim 512 --max_iter 10000 --save_model_interval 1000 --max_iter 50000

# 2022-09-24 在4个SE_10上做实验
#CUDA_VISIBLE_DEVICES=0 python train.py --style_dir /data/yuzun/MRI_data/ADNI1_x/A/train --content_dir /data/yuzun/SE_0919/SE_all_flip --save_dir /data/yuzun/Stytr_experience/012_SEall_1 --batch_size 3  --hidden_dim 512 --max_iter 300000 --save_model_interval 1000 

# 2022-10-01 在4个SE_10和272个ADNI1的sub上做实验
#CUDA_VISIBLE_DEVICES=0 python train.py --style_dir /data/yuzun/ADNI1_272sub_flip/train_flip --content_dir /data/yuzun/SE_0919/SE_all_flip --save_dir /data/yuzun/Stytr_experience/013_SEall_2 --batch_size 3  --hidden_dim 512 --max_iter 300000 --save_model_interval 1000

# 2022-10-06 在4个SE_10 SE_medium
#CUDA_VISIBLE_DEVICES=0 python train.py --style_dir /data/yuzun/SE_0919/SE_0921_reged_medium_imgs_flip --content_dir /data/yuzun/SE_0919/SE_all_flip --save_dir /data/yuzun/Stytr_experience/014_SEall_3 --batch_size 3  --hidden_dim 512 --max_iter 300000 --save_model_interval 1000

# 2022-10-08 在4个SE_10 SE_medium
#CUDA_VISIBLE_DEVICES=0 python train.py --style_dir /data/yuzun/SE_0919/SE_0921_reged_medium_imgs_flips/train --content_dir /data/yuzun/SE_0919/SE_all_flip/train --save_dir /data/yuzun/Stytr_experience/015_SEall_4 --batch_size 3  --hidden_dim 512 --max_iter 300000 --save_model_interval 1000

#2022-10-11 没有配准过的SE_medium和HR的风格学习
#CUDA_VISIBLE_DEVICES=0 python train.py --style_dir /data/yuzun/SE_0919/SE_0921_ori/MR/all --content_dir /data/yuzun/SE_0919/SE_0921_ori/HR/all --save_dir /data/yuzun/Stytr_experience/016_SEall_ori --batch_size 3  --hidden_dim 512 --max_iter 300000 --save_model_interval 1000

#2023-11-24 没有配准过的SE_medium和HR的风格学习
CUDA_VISIBLE_DEVICES=0 python train.py --style_dir /data/yuzun/SR/data/SE_0921_ori/MR/all --content_dir /data/yuzun/SR/data/SE_0921_ori/HR/all --save_dir /data/yuzun/Stytr_experience/017_SEall_ori_1 --batch_size 2  --hidden_dim 512 --max_iter 300000 --save_model_interval 5000
