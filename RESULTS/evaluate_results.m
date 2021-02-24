% This script can help you compute PSNR results

addpath('./EVAL')

dataset_path = './dt1_Bioptigen_SDOCT';
dataset_dirs = dir(dataset_path);
dataset_dirs = dataset_dirs(3:end);

% The outputs of the method are currently saved HERE
method_folder = 'PG-GMM';
output_imgs = dir(sprintf('./%s/*.tif',method_folder));
N = length(output_imgs);

% PSNRs will be saved HERE
xlsxfile = sprintf('%s_results.xlsx',method_folder);

% PSNRs are calculated in two ways:
PSNRs = zeros(N,2);          % using the whole images
PSNRs_shaved = zeros(N,2);   % using the shaved images


for i = 1:N
    
    img_dir = dataset_dirs(i).name;
    
    % High-SNR High Resolution Image (Ground Truth)
    im = imread(fullfile(dataset_path,img_dir,'average.tif'));
    im = double(im);
    
    % Noisy Test Image
    imn =  imread(fullfile(dataset_path,img_dir,'test.tif'));
    imn = double(imn);
    
    
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % Reconstructed (Denoised) Image  
    
    output_filename = output_imgs(i).name;
    
    disp([img_dir,', ',output_filename]); % corresponding Groun truth and output 
    
    output_img_path = fullfile(output_imgs(i).folder,output_filename);
    imout = imread(output_img_path);
    
    imout = double(imout);
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    PSNRs(i,1) = comp_psnr(im ,imn);
    PSNRs(i,2) = comp_psnr(im ,imout);
    
    % positions for cropping are saved here:
    load(sprintf('synthetic_average%d_roi.mat',i)) % For loading "pos" array for the i-th image.
    im_shaved=imcrop(im,pos);
    imn_shaved=imcrop(imn,pos);
    imout_shaved=imcrop(imout,pos);

    PSNRs_shaved(i,1) =comp_psnr(im_shaved ,imn_shaved);
    PSNRs_shaved(i,2) =comp_psnr(im_shaved ,imout_shaved);
    
    %figure,imshow(imout/255),title(sprintf('%.2f',PSNRs(i,2)))
    %figure,imshow(imout_shaved/255),title(sprintf('%.2f',PSNRs_shaved(i,2)))
    
end

t1 = array2table(PSNRs);
t1.Properties.VariableNames(1) = {'imn'};
t1.Properties.VariableNames(2) = {'imout'};

t2 = array2table(PSNRs_shaved);
t2.Properties.VariableNames(1) = {'imn'};
t2.Properties.VariableNames(2) = {'imout'};


writetable(t1,xlsxfile,'Sheet',1,'Range','A1');
writetable(t2,xlsxfile,'Sheet',1,'Range','D1');