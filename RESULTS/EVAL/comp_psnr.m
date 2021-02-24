% INPUTS:
% im: orginal image
% imf: reconstructed (or denoised) image
%
function [PSNR]=comp_psnr(im,imf)
MSE=mean(mean((im(:)-imf(:)).^2));
if max(im(:))<2
    MaxI=1;
else
    MaxI=255;
end
PSNR=10*log10((MaxI^2)/MSE);
