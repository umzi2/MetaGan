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
    title "B: MetaGan drop 0.2 vs G: MetaGan vs R: DUnet"
    x-axis [5k, 10k, 15k, 20k, 25k, 30k, 35k, 40k, 45k, 50k]
    y-axis "SSIM (higher is better)"
    line [0.7603138089179993, 0.7595086693763733, 0.7589496970176697]
    line [0.759844183921814, 0.7593117952346802, 0.7587045431137085, 0.7584884762763977, 0.7596063613891602, 0.7585034966468811, 0.7590800523757935, 0.7589076161384583, 0.7578854560852051, 0.759045422077179]
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
