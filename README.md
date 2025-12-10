# INR meets Multi-Contrast MRI Reconstruction

**Natascha Niessen**, Carolin M. Pirkl, Ana Beatriz Solana, Hannah
Eichhorn, Veronika Spieker, Wenqi Huang, Tim Sprenger, Marion I.
Menzel, and Julia A. Schnabel on behalf of the PREDICTOM
consortium

Accepted at [Reconstruction and Imaging Motion Estimation (RIME) Workshop at MICCAI 2025](https://rime-miccai25.github.io/) | [Link to paper](https://arxiv.org/abs/2509.04888)

**Abstract:** Multi-contrast MRI sequences allow for the acquisition of
images with varying tissue contrast within a single scan. The resulting
multi-contrast images can be used to extract quantitative information on
tissue microstructure. To make such multi-contrast sequences feasible for
clinical routine, the usually very long scan times need to be shortened e.g.
through undersampling in k-space. However, this comes with challenges
for the reconstruction. In general, advanced reconstruction techniques
such as compressed sensing or deep learning-based approaches can enable
the acquisition of high-quality images despite the acceleration.
In this work, we leverage redundant anatomical information of multi-contrast sequences to achieve even higher acceleration rates. We use
undersampling patterns that capture the contrast information located
at the k-space center, while performing complementary undersampling
across contrasts for high frequencies. To reconstruct this highly sparse
k-space data, we propose an implicit neural representation (INR) network that is ideal for using the complementary information acquired
across contrasts as it jointly reconstructs all contrast images. We demonstrate the benefits of our proposed INR method by applying it to multi-
contrast MRI using the MPnRAGE sequence, where it outperforms the
state-of-the-art parallel imaging compressed sensing (PICS) reconstruction method, even at higher acceleration factors.

**Keywords:** Implicit Neural Representation · MRI Reconstruction · Multi-
Contrast MRI · Quantitative MRI

## Citation
If you use this code, please cite our paper:

```bash
@misc{niessen2025inrmeetsmulticontrastmri,
      title={INR meets Multi-Contrast MRI Reconstruction}, 
      author={Natascha Niessen and Carolin M. Pirkl and Ana Beatriz Solana and Hannah Eichhorn and Veronika Spieker and Wenqi Huang and Tim Sprenger and Marion I. Menzel and Julia A. Schnabel},
      year={2025},
      eprint={2509.04888},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2509.04888}, 
}
```

## Steps to reconstruct your own multi-contrast data

1. Create a virtual environment with the required packages:
   ```bash
   conda env create -f inr_env.yml
   source activate inr_env OR conda activate inr_env
   ```
   If this fails, set up the environment from scratch using the commands in environment_setup_from_scratch.md.
   
2. Load your data at the top of the **main.py** script. The data is needed in the following format:
   
    Vx,Vy,Vz: spatial dimensions  
    C: coil dimension  
    N: contrast dimension  

    Coil Sensitivity Maps: csm (Vx, Vy, Vz, C)  
    Binary brain mask: brain_mask (Vx, Vy, Vz)  
    K-space data: kspace_loaded (Vx, Vy, Vz, C, N)  
    Reference Image (e.g. fully sampled inverse FFT reconstruction): reference_img (Vx, Vy, Vz, N)  
   
3. Select the undersampling configuration in **configs/config_direct_inr.yaml**.
   
4. Run **main.py**. The last epoch of the self-supervised training serves as inference. The results are saved in **results/direct_inr**.


## Illustration of the Multi-Contrast INR Reconstruction framework with complementary undersampling:

![Image](https://github.com/nataschaniessen/Multi-contrast_INR_MICCAI2025/blob/main/Multi-Contrast%20INR%20Reconstruction.png)
   
