cd ./src2
CUDA_VISIBLE_DEVICES=1 python main.py --fitness latency --input_size 64 --model resnet18 --num_pe 65536 --l1_size 65536 --l2_size 1048576 --bandwidth 4096 --epochs 10
cd ../..