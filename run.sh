rm -rf runs/*MNIST
python train_ibn_dcgan.py --epochs=150 --g-learning-rate=1e-3 --eg-learning-rate=1e-3 --d-learning-rate=1e-4 --latent-size=2