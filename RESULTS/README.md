
# OCT Denoising Results
OCT denoising results of the proposed [MIFCN method](https://github.com/ashkan-abbasi66/MIFCN) and the compared methods<br> The outputs of each method is saved in a folder entitled `<method>`. E.g., `BM3D`<br>
# Compared methods
The proposed MIFCN method is compared with some of the well-known state-of-the-art denoising methods from the literature. The comparison methods include: K-SVD denoising algorithm [1], BM3D [2], SAIST [3], PG-GMM [4], BM4D [5], and SSR [6].

# Figures of Visual Results in the paper
Click on the images to see/download the original high-quality images.<br>
<a href="https://github.com/ashkan-abbasi66/MIFCN_Results/blob/master/Figure%203-visual%20results_2.5_with13.png">
<img src="https://github.com/ashkan-abbasi66/MIFCN_Results/blob/master/Figure%203-low%20quality.jpg" width="306" height="389" class="center"/>
</a>
<br>
Fig. 3: Visual comparison of two denoised images by the compared methods. First column: (A) Original Noisy Image; (B) KSVD Denoising (PSNR = 26.05); (C) BM3D (PSNR = 26.25); (D) SAIST (PSNR = 26.01); (E) PG-GMM (PSNR = 26.1); (F) BM4D (PSNR = 26.48); (G) SSR (PSNR = 26.89); (H) The proposed MIFCN method (PSNR = 27.49); (I) The registered and averaged images. Second column: (J) Original Noisy Image; (K) KSVD Denoising (PSNR = 26.13); (L) BM3D (PSNR = 26.02); (M) SAIST (PSNR = 26.16); (N) PG-GMM (PSNR = 25.89); (O) BM4D (PSNR = 26.54); (P) SSR (PSNR = 27.06); (Q) The proposed MIFCN method (PSNR = 27.56); (R) The registered and averaged images.
<br>
<!-- <a href="https://github.com/ashkan-abbasi66/MIFCN_Results/blob/master/Figure%204-more%20visual%20results_2.5_better.png">
<img src="https://github.com/ashkan-abbasi66/MIFCN_Results/blob/master/Figure%204-low%20quality.jpg" width="306" height="389" class="center"/>
</a>
<br>
Fig. 4: More visual results for retinal OCT image denoising. First column: (A) Original Noisy Image; (B) KSVD Denoising (PSNR = 27.22); (C) BM3D (PSNR = 27.30); (D) SAIST (PSNR = 27.24); (E) PG-GMM (PSNR = 27.33); (F) BM4D (PSNR = 27.63); (G) SSR (PSNR = 27.61); (H) The proposed MIFCN method (PSNR = 28.53); (I) The registered and averaged images. Second column: (J) Original Noisy Image; (K) KSVD Denoising (PSNR = 21.90); (L) BM3D (PSNR = 21.65); (M) SAIST (PSNR = 21.72); (N) PG-GMM (PSNR = 21.68); (O) BM4D (PSNR = 21.78); (P) SSR (PSNR = 22.06); (Q) The proposed MIFCN method (PSNR = 22.43); (R) The registered and averaged images. -->

# Computing PSNR metric
The PSNR results that are reported in the table are obtained after shaving output and ground-truth images. This is done because we want to focus on the important parts of each OCT image. Some algorithms may have inferior results in recovering layers and textures but might have good ability to remove noise in smooth areas. Therefore, to compute PSNR metric, the output images of all algorithms are shaved. You can use the following piece of code to replicate the results that are reported in the paper.

```matlab
% Truth: Ground-truth image
% im_out: output of denoising algorithm

load(sprintf('synthetic_average%d_roi.mat',i)) % For loading "pos" array for the i-th image.
Truth_shaved=imcrop(Truth,pos);
im_out_shaved=imcrop(im_out,pos);

PSNR=comp_psnr(Truth_shaved ,im_out_shaved);
```
The ROIs and the `comp_psnr` function are saved into `EVAL` folder.

**IMPORTANT UPDATE at 2/24/2021** 
  - Just use `evaluate_results.m` to compute PSNR for the saved results. It computes the PSNRs for all images and then save them in an excel file. The PSNRs are computed in two ways: 1) using the whole ground-truth and output images, 2) using the shaved (cropped) ground-truth and output images. 
  - We have just noticed that the PSNR results reported below fig. 3 are belong to the outputs of the methods for image #11 in the dataset, while the figure in the paper shows the results corresponding to image #3. Sorry for the inconvenience that may have caused for you!

# References
[1] M. Elad and M. Aharon, “Image Denoising Via Sparse and Redundant Representations Over Learned Dictionaries,” IEEE Trans. Image Process., vol. 15, no. 12, pp. 3736–3745, Dec. 2006.<br>
[2] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, “Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering,” IEEE Trans. Image Process., vol. 16, no. 8, pp. 2080–2095, Aug. 2007.<br>
[3] W. Dong, G. Shi, and X. Li, “Nonlocal image restoration with bilateral variance estimation: A low-rank approach,” IEEE Trans. Image Process., vol. 22, no. 2, pp. 700–711, 2013.<br>
[4] J. Xu, L. Zhang, W. Zuo, D. Zhang, and X. Feng, “Patch Group Based Nonlocal Self-Similarity Prior Learning for Image Denoising,” in
Proceedings of the IEEE International Conference on Computer Vision, 2015, pp. 244–252.<br>
[5] M. Maggioni, V. Katkovnik, K. Egiazarian, and A. Foi, “Nonlocal Transform-Domain Filter for Volumetric Data Denoising and Reconstruction,” IEEE Trans. Image Process., vol. 22, no. 1, pp. 119–133, Jan. 2013.<br>
[6] L. Fang, S. Li, D. Cunefare, and S. Farsiu, “Segmentation Based Sparse Reconstruction of Optical Coherence Tomography Images,” IEEE Trans. Med. Imaging, vol. 36, no. 2, pp. 407–421, Feb. 2017.<br>
