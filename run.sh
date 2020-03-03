rm -rf runs/*MNIST
python train_ibn_dcgan.py --epochs=150 --g-learning-rate=2e-4 --eg-learning-rate=2e-4 --d-learning-rate=2e-4