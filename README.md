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

1. Sử dụng kiến trúc **One-Stream Transformer** giúp giảm thời gian huấn luyện và suy luận.
2. Kết hợp trực tiếp **template** và **search region** trong một lần truyền qua backbone.
3. Tích hợp cơ chế **Early Candidate Elimination (ECE)** giúp giảm số lượng token cần xử lý, từ đó giảm chi phí tính toán và bộ nhớ.
4. Sử dụng trọng số **MAE ViT-Base** được huấn luyện trước giúp mô hình hội tụ nhanh và đạt hiệu quả cao.
5. Được tùy chỉnh để huấn luyện và đánh giá trên bộ dữ liệu **Anti-UAV410**.
## Cài đặt môi trường 

Sử dụng Anaconda (CUDA 11.3)
```
conda env create -f ostrack_cuda113_env.yaml
```


## Thiết lập đường dẫn 
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

## Chuẩn bị data
Link tải data: https://drive.google.com/file/d/1zsdazmKS3mHaEZWS2BnqbYHPEcIaH5WR/view
Tải về, giải nén và đặt data vào cấu trúc như hình
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
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output 
```

## Kiểm thử 
Tải model weight [Google Drive](https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd?usp=sharing) 

Đặt model weights ở `$PROJECT_ROOT$/output/checkpoints/train/ostrack`


```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset antiuav410_test
python tracking/analysis_results.py # need to modify tracker configs and names
```

## Hiển thị
1. Mở một terminal khác và chạy visdom

2. Chạy lệnh 
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset antiuav410_test --threads 1 --num_gpus 1 --debug 1
```
3. Mở `http://localhost:8097` 

4. Kết quả hiển thị

![ECE_vis](https://github.com/botaoye/OSTrack/blob/main/assets/vis.png)


## Kiêm tra tốc độ 

```
# Profiling vitb_256_mae_ce_32x4_ep300
python tracking/profile_model.py --script ostrack --config vitb_256_mae_ce_32x4_ep300
# Profiling vitb_384_mae_ce_32x4_ep300
python tracking/profile_model.py --script ostrack --config vitb_384_mae_ce_32x4_ep300
```


