rm -rf runs/*MNIST
rm -rf runs/*IBN*

# python train_ibn_dcgan.py --epochs=150 --dataset-name=CelebA --g-learning-rate=2e-4 --eg-learning-rate=2e-4 --d-learning-rate=2e-4 --data-dir=/ndata/CelebA/


python train_ibn_dcgan.py --epochs=150 --dataset-name=MNIST --g-learning-rate=2e-4 --eg-learning-rate=2e-4 --d-learning-rate=2e-4