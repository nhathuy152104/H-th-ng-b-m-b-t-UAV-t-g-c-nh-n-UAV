# Hệ thống bám bắt UAV từ góc nhìn UAV

<p align="center">
  <img width="85%" src="https://github.com/botaoye/OSTrack/blob/main/assets/arch.png" alt="Framework"/>
</p>


### :star2: Giới thiệu
Dự án xây dựng hệ thống bám bắt UAV từ góc nhìn của một UAV khác (UAV-to-UAV Tracking) dựa trên mô hình OSTrack (One-Stream Tracker).

Khác với các bộ theo dõi Siamese truyền thống, OSTrack kết hợp trực tiếp template và search region thành một chuỗi đầu vào duy nhất cho Vision Transformer, giúp mô hình học đồng thời đặc trưng và quan hệ giữa hai vùng ảnh. Nhờ đó, hệ thống đạt được tốc độ suy luận cao trong khi vẫn duy trì độ chính xác tốt.

Trong đề tài này, mô hình được huấn luyện và đánh giá trên bộ dữ liệu Anti-UAV410, hướng tới bài toán theo dõi UAV trong điều kiện:

1. UAV mục tiêu có kích thước nhỏ.
2. Góc nhìn thay đổi liên tục.
3. Chuyển động nhanh.
4. Mất mục tiêu tạm thời.
5. Nền phức tạp và nhiều nhiễu.

### :star2: Đặc điểm nổi bật 
OSTrack-256 can be trained in ~24 hours with 4*V100 (16GB of memory per GPU), which is much faster than recent SOTA transformer-based trackers. The fast training speed comes from:

1. While previous Siamese-style trackers required separate feeding of the template and search region into the backbone at each iteration of training, OSTrack directly combines the template and search region. The tight and highly parallelized structure results in improved training and inference speed.
  
2. The proposed early candidate elimination (ECE) module significantly reduces memory and time consumption.
  
3. Pretrained Transformer weights enable faster convergence.

### :star2: Good performance-speed trade-off

[//]: # (![speed_vs_performance]&#40;https://github.com/botaoye/OSTrack/blob/main/assets/speed_vs_performance.png&#41;)
<p align="center">
  <img width="70%" src="https://github.com/botaoye/OSTrack/blob/main/assets/speed_vs_performance.png" alt="speed_vs_performance"/>
</p>

## Cài đặt môi trường 

Sử dụng Anaconda (CUDA 11.3)

conda env create -f ostrack_cuda113_env.yaml



## Thiết lập đường dẫn 
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

## Data Preparation
Put the tracking datasets in ./data. It should look like this:
   ```
   ${PROJECT_ROOT}
    -- data
        -- AntiUAV410
            |-- train
            |-- test
            |-- val
            ...

   ```


## Huấn luyện 
Tải pretrained backbone [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) và đặt vào `$PROJECT_ROOT$/pretrained_models` 
```
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

## Kiểm thử 
Tải model weight [Google Drive](https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd?usp=sharing) 

Đặt model weights ở `$PROJECT_ROOT$/output/checkpoints/train/ostrack`


```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset antiuav410_test
python tracking/analysis_results.py # need to modify tracker configs and names
```

## Visualization or Debug 
[Visdom](https://github.com/fossasia/visdom) is used for visualization. 
1. Mở một terminal khác và chạy visdom

2. Chạy lệnh 
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset antiuav410_test --threads 1 --num_gpus 1 --debug 1
```
3. Chạy lệnh `http://localhost:8097` 

4. Kết quả hiển thị

![ECE_vis](https://github.com/botaoye/OSTrack/blob/main/assets/vis.png)


## Kiêm tra tốc độ 

```
# Profiling vitb_256_mae_ce_32x4_ep300
python tracking/profile_model.py --script ostrack --config vitb_256_mae_ce_32x4_ep300
# Profiling vitb_384_mae_ce_32x4_ep300
python tracking/profile_model.py --script ostrack --config vitb_384_mae_ce_32x4_ep300
```


