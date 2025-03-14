#!/bin/bash

python3 trainer.py --exp_name 50cifar --dataset cifar10 --config hyperparameters/cifar10/50cifar10.yaml --device cuda:2
python3 trainer.py --exp_name 51cifar --dataset cifar10 --config hyperparameters/cifar10/51cifar10.yaml --device cuda:2
python3 trainer.py --exp_name 52cifar --dataset cifar10 --config hyperparameters/cifar10/52cifar10.yaml --device cuda:2
python3 trainer.py --exp_name 53cifar --dataset cifar10 --config hyperparameters/cifar10/53cifar10.yaml --device cuda:2
python3 trainer.py --exp_name 54cifar --dataset cifar10 --config hyperparameters/cifar10/54cifar10.yaml --device cuda:2



python3 trainer.py --exp_name 55cifar --dataset cifar10 --config hyperparameters/cifar10/55cifar10.yaml --device cuda:2
# python3 trainer.py --exp_name 56cifar --dataset cifar10 --config hyperparameters/cifar10/56cifar10.yaml --device cuda:1
# python3 trainer.py --exp_name 57cifar --dataset cifar10 --config hyperparameters/cifar10/57cifar10.yaml --device cuda:2

# python3 trainer.py --exp_name 58cifar --dataset cifar10 --config hyperparameters/cifar10/58cifar10.yaml --device cuda:1
# python3 trainer.py --exp_name 59cifar --dataset cifar10 --config hyperparameters/cifar10/59cifar10.yaml --device cuda:1
