# DDPG policy network

`DDPG_Policy_Network_FULL.ipynb` trains a DDPG agent that allocates bandwidth and power across the videos watched by a crowd of headsets, using the frozen FoV predictors and the CVAE-GAN encoder from this repository.

## Run
```
pip install -r ../requirements.txt jupyterlab
cd ddpg_policy_network
jupyter lab
```
Open the notebook and Run All. Start Jupyter from this folder: the notebook finds the data at `../dataset/360dataset/sensory/tile` and reads and writes `./outputs_predictor_vae_combined/`.

## Inputs
- `../dataset/360dataset/sensory/tile/`: viewer tile maps for coaster, coaster2, diving, drive and game
- `outputs_predictor_vae_combined/cvaegan_encoder.pt`: same file as `../vae/cvaegan_encoder.pt`
- `outputs_predictor_vae_combined/predictor_<video>.pt`: one FoV predictor per video

## What it trains
With `RETRAIN = True` it trains from scratch: 6 trainings of 6000 episodes. The discount sweep (0.9, 0.95, 0.99) runs on the 8K-20MHz scenario, then the best discount is reused on 8K-10MHz, 4K-20MHz and 4K-10MHz. It prints a time estimate after 50 episodes. It runs on CPU, CUDA or Apple MPS.
