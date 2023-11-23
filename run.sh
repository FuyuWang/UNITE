cd ./src
CUDA_VISIBLE_DEVICES=0 python main.py --fitness latency --input_size 64 --model resnet18 --num_pe 8192 --l1_size 65536 --l2_size 1048576 --bandwidth 4096 --epochs 10
cd ../..