#CUDA_VISIBLE_DEVICES=0,1  python test.py --style_dir Image/ --content_dir /data/yuzun/MRI_data/hcp_all/flip/val --output /data/yuzun/StyTR_out_val --decoder_path /data/yuzun/StyTR-2_model_320/decoder_iter_10000.pth --Trans_path /data/yuzun/StyTR-2_model_320/transformer_iter_10000.pth --embedding_path /data/yuzun/StyTR-2_model_320/embedding_iter_10000.pth --hidden_dim 320
# 1
#CUDA_VISIBLE_DEVICES=0,1  python test.py --style_dir Image/ --content_dir /data/yuzun/MRI_data/hcp_all/stytr_test --output /data/yuzun/7_19_stytr_results/StyTR-2_model_320 --decoder_path /data/yuzun//Stytr_experience/StyTR-2_model_320/decoder_iter_10000.pth --Trans_path /data/yuzun//Stytr_experience/StyTR-2_model_320/transformer_iter_10000.pth --embedding_path /data/yuzun//Stytr_experience/StyTR-2_model_320/embedding_iter_10000.pth --hidden_dim 320
# 004
#CUDA_VISIBLE_DEVICES=0,1  python test.py --style_dir Image/ --content_dir /data/yuzun/MRI_data/hcp_all/stytr_test --output /data/yuzun/7_19_stytr_results/004_StyTR-2_model_content10_200000 --decoder_path /data/yuzun/Stytr_experience/004_StyTR-2_model_content10/decoder_iter_200000.pth --Trans_path /data/yuzun/Stytr_experience/004_StyTR-2_model_content10/transformer_iter_200000.pth --embedding_path /data/yuzun/Stytr_experience/004_StyTR-2_model_content10/embedding_iter_200000.pth --hidden_dim 320
#CUDA_VISIBLE_DEVICES=0,1  python test.py --style_dir Image/ --content_dir /data/yuzun/MRI_data/hcp_all/stytr_test --output /data/yuzun/7_19_stytr_results/005_StyTR-2_model_content15_190000 --decoder_path /data/yuzun/Stytr_experience/005_StyTR-2_model_content15/decoder_iter_190000.pth --Trans_path /data/yuzun/Stytr_experience/005_StyTR-2_model_content15/transformer_iter_190000.pth --embedding_path /data/yuzun/Stytr_experience/005_StyTR-2_model_content15/embedding_iter_190000.pth --hidden_dim 320

#006
#CUDA_VISIBLE_DEVICES=0,1  python test.py --style_dir Image/ --content_dir /data/yuzun/MRI_data/hcp_all/stytr_test --output /data/yuzun/7_19_stytr_results/006_StyTR-2_model_style10_200000 --decoder_path /data/yuzun/Stytr_experience/006_StyTR-2_model_style10/decoder_iter_200000.pth --Trans_path /data/yuzun/Stytr_experience/006_StyTR-2_model_style10/transformer_iter_200000.pth --embedding_path /data/yuzun/Stytr_experience/006_StyTR-2_model_style10/embedding_iter_200000.pth --hidden_dim 320

# 9-21 test 0919的SE_10
#CUDA_VISIBLE_DEVICES=0  python test.py --style_dir Image --content_dir /data/yuzun/SE_0919/SE_0919_HR_flip --output ~/stytr_011_SE_0919 --decoder_path /data/yuzun/Stytr_experience/011_SE919_1/decoder_iter_10000.pth --Trans_path /data/yuzun/Stytr_experience/011_SE919_1/transformer_iter_10000.pth --embedding_path /data/yuzun/Stytr_experience/011_SE919_1/embedding_iter_10000.pth --hidden_dim 512

# 9-25 test SE_all, 注意content,style size都是256，因为512装不下。。
#CUDA_VISIBLE_DEVICES=0  python test.py --style_dir Image --content_dir /data/yuzun/SE_0919/SE_all_flip_2/train --output /data/yuzun/Stytr_11_SE10all_out --decoder_path /data/yuzun/Stytr_experience/012_SEall_1/decoder_iter_300000.pth --Trans_path /data/yuzun/Stytr_experience/012_SEall_1/transformer_iter_300000.pth --embedding_path /data/yuzun/Stytr_experience/012_SEall_1/embedding_iter_300000.pth --hidden_dim 512

# 10-3日 test 用扩充后的ADNI1的数据集,使用以前的Image作为style
#CUDA_VISIBLE_DEVICES=0  python test.py --style_dir Image --content_dir /data/yuzun/SE_0919/SE_all_flip_2/train --output /data/yuzun/Stytr_013_SEall_2_out_Image --decoder_path /data/yuzun/Stytr_experience/013_SEall_2/decoder_iter_300000.pth --Trans_path /data/yuzun/Stytr_experience/013_SEall_2/transformer_iter_300000.pth --embedding_path /data/yuzun/Stytr_experience/013_SEall_2/embedding_iter_300000.pth --hidden_dim 512

# 10-3 从训练集中挑选出来的Image_2
#CUDA_VISIBLE_DEVICES=0  python test.py --style_dir Image_2 --content_dir /data/yuzun/SE_0919/SE_all_flip_2/train --output /data/yuzun/Stytr_013_SEall_2_out_Image_2 --decoder_path /data/yuzun/Stytr_experience/013_SEall_2/decoder_iter_300000.pth --Trans_path /data/yuzun/Stytr_experience/013_SEall_2/transformer_iter_300000.pth --embedding_path /data/yuzun/Stytr_experience/013_SEall_2/embedding_iter_300000.pth --hidden_dim 512

# 10-12 test stytr
#content_path='/data/yuzun/SE_0919/SE_0921_ori/HR/test'
#style_path='/data/yuzun/SE_0919/SE_0921_ori/MR/test'
## 记录nii文件的数量
#count=0
#
#cd $content_path
#pwd
#for subject in `ls`
#do
##  echo $content_path/$subject
#  stylesub=${subject:0:-4}m.png
##  echo $style_path/$stylesub
#  CUDA_VISIBLE_DEVICES=0  python /home/yuzun/project/MRI_multiModel_SR/StyTR-2/test.py --style_dir $style_path/$stylesub --content_dir $content_path/$subject --output /data/yuzun/SE_0919/SE_0921_ori/stytr --decoder_path /data/yuzun/Stytr_experience/016_SEall_ori/decoder_iter_142000.pth --Trans_path /data/yuzun/Stytr_experience/016_SEall_ori/transformer_iter_142000.pth --embedding_path /data/yuzun/Stytr_experience/016_SEall_ori/embedding_iter_142000.pth --hidden_dim 512
#done

# 10-12 test stytr 一张图片
#CUDA_VISIBLE_DEVICES=0  python test.py --style_dir Image_3 --content_dir /data/yuzun/SE_0919/SE_0921_ori/HR/test --output /data/yuzun/Stytr_014_SEall_2_out_Image_3 --decoder_path /data/yuzun/Stytr_experience/016_SEall_ori/decoder_iter_159000.pth --Trans_path /data/yuzun/Stytr_experience/016_SEall_ori/transformer_iter_159000.pth --embedding_path /data/yuzun/Stytr_experience/016_SEall_ori/embedding_iter_159000.pth --hidden_dim 512

# 11-1 test stytr
#CUDA_VISIBLE_DEVICES=0  python test.py --style_dir Image_3 --content_dir /data/yuzun/SE_0919/SE_0921_ori/HR/test --output /data/yuzun/Stytr_014_SEall_2_out_Image_4 --decoder_path /data/yuzun/Stytr_experience/016_SEall_ori/decoder_iter_300000.pth --Trans_path /data/yuzun/Stytr_experience/016_SEall_ori/transformer_iter_300000.pth --embedding_path /data/yuzun/Stytr_experience/016_SEall_ori/embedding_iter_300000.pth --hidden_dim 512

# 11-2 将size改成512
#CUDA_VISIBLE_DEVICES=0  python test.py --style_dir Image_3 --content_dir /data/yuzun/SE_0919/SE_0921_ori/HR/test --output /data/yuzun/Stytr_014_SEall_2_out_Image_5 --decoder_path /data/yuzun/Stytr_experience/016_SEall_ori/decoder_iter_300000.pth --Trans_path /data/yuzun/Stytr_experience/016_SEall_ori/transformer_iter_300000.pth --embedding_path /data/yuzun/Stytr_experience/016_SEall_ori/embedding_iter_300000.pth --hidden_dim 512

# 11-29 针对benchmark的test
CUDA_VISIBLE_DEVICES=0  python test.py --style /data/yuzun/SR/data/SE_0921_ori/MR/test/SE_0921_3_109m.png --content_dir /data/yuzun/SR/data/SE_0921_ori/HR/test --output /data/yuzun/SR/results/benchmark --decoder_path /data/yuzun/SR/experience/benchmark/decoder_iter_300000.pth --Trans_path /data/yuzun/SR/experience/benchmark/transformer_iter_300000.pth --embedding_path /data/yuzun/SR/experience/benchmark/embedding_iter_300000.pth --hidden_dim 512
