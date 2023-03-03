python3 generation.py --data_dir data --percent 5pct --model MLP --batch_size 128 --mitigation andmask --lr 1e-5 --epoch 150 --logs andmask/logs_1 --lambda_penalty 0.8

python3 generation.py --data_dir data --percent 5pct --model MLP --batch_size 128 --mitigation andmask --lr 1e-4 --epoch 150 --logs andmask/log_2  --lambda_penalty 0.8


python3 generation.py --data_dir data --percent 5pct --model MLP --batch_size 128 --mitigation andmask --lr 1e-5 --epoch 150 --logs andmask/log_4  --lambda_penalty 0.5


python3 generation.py --data_dir data --percent 5pct --model MLP --batch_size 128 --mitigation andmask --lr 1e-4 --epoch 150 --logs andmask/logs_5 --lambda_penalty 0.5

python3 generation.py --data_dir data --percent 5pct --model MLP --batch_size 128 --mitigation andmask --lr 1e-5 --epoch 150 --logs andmask/log_6  --lambda_penalty 0.2

python3 generation.py --data_dir data --percent 5pct --model MLP --batch_size 128 --mitigation andmask --lr 1e-5 --epoch 150 --logs andmask/log_7  --lambda_penalty 0.2


python3 generation.py --dataset celeba --data_dir data --percent 5pct --model resnet18 --batch_size 128 --mitigation lff --lr 1e-5 --epoch 5  --logs logs --lambda_penalty 0.8
