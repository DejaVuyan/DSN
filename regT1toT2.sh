#!/bin/bash
# add by yuzun 2021/11/2
pairedrootPath='/data/yuzun/MRI_data/reged_ADNI/paired_data'
outrootPath='/data/yuzun/MRI_data/reged_ADNI/reged_data'
# 记录nii文件的数量
count=0

cd $pairedrootPath
pwd
for subject in `ls`
do 
#    echo $subject
    cd $subject
    # 进入T1文件夹读取nii文件
    cd T1  # 变量才是红色的，这里不是变量
    for T1_nii in `ls`
    do 
#        echo $nii
        #awk 'BEGIN{print "'$nii'"}'
        #awk -v niiname=$nii 'BEGIN{FS="_";print niiname}{print $0}'  # awk赋值外部变量
        T1_time=`echo $T1_nii |cut -d_ -f 12`  # 以_为分隔符，第12列赋值给nii_time
        T1_time=${T1_time:0:4}   # 截取字符串的前4个字符赋值给nii_time
        
        # 读取T2 MRI
        cd ../T2
        for T2_nii in `ls`
        do 
            T2_time=`echo $T2_nii |cut -d_ -f 9`
            T2_time=${T2_time:0:4}
            if [ $T1_time = $T2_time ];then  # 字符串判断用一个等号
                # ！！！找到匹配nii文件，在这个里面操作
                let count=$count+1   # 要加上let，不然就是个字符串
                T1_path=$pairedrootPath/$subject/T1/$T1_nii
                T2_path=$pairedrootPath/$subject/T2/$T2_nii
                
                out_time_path=$outrootPath/$subject/$T1_time
                mkdir -p $out_time_path
                # 将T2nii文件移动到输出目录中去
                cp $T2_path $out_time_path/$T2_nii

                out_filename=$T1_nii
                out_filename=`echo $out_filename |cut -d. -f 1`
                # 更改过的输出文件名
                out_filename=$out_filename'regtoT2.nii'
                # 完整路径
                out_path=$out_time_path/$out_filename
                echo "T1 MRI path is "$T1_path
                echo "T2 MRI path is "$T2_path
                /usr/local/fsl/bin/flirt -in $T1_path -ref $T2_path -out $out_path -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear
                echo "output_path is     "$out_path
                echo $count

                # 空一行
                echo 
            fi
        done
    done
    # 最后别忘了回到rootpath
    cd $pairedrootPath
done
