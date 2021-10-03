close all
clc
clear all
[filename,pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.gif';'*.png'},'Pick an Image File');
I = imread([pathname,filename]);
figure, imshow(I); title('Blood cell Image');
I = imresize(I,[200,200]);

% Convert to grayscale
gray = rgb2gray(I);

% Otsu Binarization for segmentation
level = graythresh(I);
img = im2bw(I,level);
figure, imshow(img);title('Otsu Thresholded Image');

%% K means Clustering to segment 

cform = makecform('srgb2lab');
% Apply the colorform
lab_he = applycform(I,cform);


ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 1;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',1);

pixel_labels = reshape(cluster_idx,nrows,ncols);

segmented_images = cell(1,3);

rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end


figure, imshow(segmented_images{1});title('Objects in Cluster 1');

seg_img = im2bw(segmented_images{1});
figure, imshow(seg_img);title('Segmented cancer region');

%% features extraction


disp('GLCM Features are...');
g = graycomatrix(seg_img);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast
Correlation = stats.Correlation
Energy = stats.Energy
Homogeneity = stats.Homogeneity
feat = [Contrast,Correlation,Energy,Homogeneity];


%% classification
load datab.mat
 xdata = measure;
 group = la;
 data = measure;
groups = ismember(la,'Normal');
groups = ismember(la,'Abnormal');
[train,test] = crossvalind('HoldOut',groups);
cp = classperf(groups);
svmStruct = svmtrain(data(train,:),groups(train),'showplot',false,'kernel_function','linear');
classes = svmclassify(svmStruct,data(test,:),'showplot',false);
classperf(cp,classes,test);
Accuracy_Classification = cp.CorrectRate.*100;
sprintf('Accuracy of Linear kernel is: %g%%',Accuracy_Classification)

 svmStruct1 = svmtrain(xdata,group,'kernel_function', 'linear');
 
 species = svmclassify(svmStruct1,feat,'showplot',false)