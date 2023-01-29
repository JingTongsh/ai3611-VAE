set -exuo pipefail

python main.py --latent_dim 1 --cuda
python main.py --latent_dim 2 --cuda
python main.py --latent_dim 8 --cuda

python main.py --latent_dim 1 --mse_loss --cuda
python main.py --latent_dim 2 --mse_loss --cuda
python main.py --latent_dim 8 --mse_loss --cuda
