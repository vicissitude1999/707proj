
# Train VAE
python src/train_VAE.py tools/train_vae_cifar10.json

# Train VAEBM

Specify the checkpoint argument in train_vaebm_cifar10.json, and then run

python src/train_VAEBM.py tools/train_vaebm_cifar10.json

python -m pytorch_fid /data/10707project/output/groundtruth /data/10707project/output/generated/beta_1/random --device cuda:1
python -m pytorch_fid /data/10707project/output/groundtruth /data/10707project/output/generated/beta_2/mulog_added --device cuda:1
python -m pytorch_fid /data/10707project/output/groundtruth /data/10707project/output/generated/beta_2/mulog_mcmc --device cuda:1


# compare fid of randomly generated images between VAE and VAEBM/EBM
python -m pytorch_fid /data/10707project/output/groundtruth /data/10707project/output/vae/celeba64/beta_1/random --device cuda:0

python -m pytorch_fid /data/10707project/output/groundtruth /data/10707project/output/vae/celeba64/beta_1/recon --device cuda:0


random samples
         VAE      VAEBM
beta=1   75.571   75.591
beta=2   79.821   73.580
beta=4   88.479   83.063
beta=10  107.176  96.277

reconstructions
         VAE      VAEBM (mulog added)   mulog mcmc      mulog both mcmc
beta=1   50.408   50.790                52.668          43.855
beta=2   60.915   60.978                58.621          53.634
beta=4   74.754   75.061                71.409          61.221
beta=10  98.109   98.484                90.154          85.666