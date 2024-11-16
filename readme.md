# MetaGan.
This project was developed based on the idea that for the GAN loss function, only a single value is needed—indicating either high or super resolution (HR or SR). Therefore, an attempt was made to implement the loss function using binary classification. As a result, this approach proved to be viable, increasing network stability and enabling the use of more advanced techniques to enhance discriminator performance, with only a minimal increase in memory and computational costs.

## Config:

net_g: Span

manual_seed: 1234

loss: perceptual(vgg19) * 0.035 + Mssim_l1_alpha_0.1 + Gan*0.1

train_dataset: df2k

val_dataset: urban100

train_framework: NeoSR

```mermaid
xychart-beta
    title "B: MetaGan conv vs G: MetaGan attn vs R: DUnet"
    x-axis [5k, 10k, 15k, 20k, 25k, 30k, 35k, 40k, 45k, 50k]
    y-axis "SSIM (higher is better)"
    line [0.7609441876411438, 0.7476010322570801, 0.7431791424751282, 0.7488889694213867, 0.7449856996536255, 0.746844470500946, 0.7367813587188721, 0.7438730597496033, 0.7396436333656311, 0.7364404797554016]
    line [0.7695407867431641, 0.7703334093093872, 0.7713731527328491, 0.7664491534233093, 0.7608891725540161, 0.7598246335983276, 0.759772002696991, 0.7588373422622681, 0.7553575038909912, 0.7695077061653137]
    line [0.7592670917510986, 0.7575504779815674, 0.7543612718582153, 0.7510430216789246, 0.7516831755638123, 0.7493809461593628, 0.7491888999938965, 0.7482668161392212, 0.746341347694397, 0.7475400567054749]

```
## MetaGan:
- In Shape: [2, 3, 512, 512]
- Out Shape: [2, 1]
- Mean iter time: 15.02ms
- Max Memory: 1275.43[M]
- Parameters: 8041.35K
## DUnet:
- In Shape: [2, 3, 512, 512]
- Out Shape: [2, 1, 512, 512]
- Mean iter time: 58.57ms
- Max Memory: 3385.33[M]
- Parameters: 3231.46K
