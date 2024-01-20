
clear
clc
% warning off
format short

%%% Loading raw dataset
%%% Select folder that contains raw holograms
Raw_holo_path = uigetdir('H:\Experiments\Spiked cell DHM-sheath_ML\Pdhm setup\');                    % Select folder where object holograms are stored
%%%
cd(Raw_holo_path)

%%
N = 800;                % Number of pixels in FOV in x-direction
M = N;                  % Number of pixels in FOV in y-direction
X = 20;                 % Pixel pitch
Y = 20;
power = 20;             % Objective magnification
di = 10;                % Set initial distance for reconstruction, this is the first plane of reconstruction
p = 125;                % Number of reconstruction planes
depth = 5;              % Distance in um between two reconstruction planes, has to be < size of individual object
% dx = X/power;         % Actual pixel size in hologram plane
dx = 1;
% dy = Y/power;
dy = dx;
Area = (N*dx);          % Dimension of side length in um
Lambda = 0.635;         % Wave number
dr1 = di;               % Wavelength of He-Ne laser emission
K = 2*pi/Lambda;        % First plane of reconstruction
dr2 = di + (p-1)*depth; % Final plane of reconstruction
HoloNum = 7000;        % Number of holograms to be used for analysis
nums = 1;              % Number of random numbers to be generated
R = 5;                  % Window size for correcting detected centroids at brightest pixel
k_thresh = 2;           % Adaptive threshold cut-off
sz_windthresh = 51;     % Window size for adaptive thresholding
el1_seg = 3;            % Erosion size for refining thresholded image
el2_seg = 5;            % Dilation size for refininf thresholded image
el1_size = 3;           % Size for morphological operations for size detection
nr = 25;                % Particles within this pixelated area not considered for analysis
nr_y = 125;              % considering only sample region due to sheath
W1 = 20;                % Window size around detected centroid for size detection
W11 = 35;               % Window size for library generation
W2 = W1+1;
W3 = 2*W1+1;
num = 5;                % Number of pixels to either side of detected centroid for fingerprinting

%%% Cleaning raw holograms
I2_seq = zeros(N);
for k = 6001:6001 + HoloNum
    filename = ['MCF7_C_10_20X_420fps_texp_35us_Z_exp_200um_' num2str(k,'%05d')];
    I1_seq = double(imread([filename '.tif']));                                  % rotating image
    I2_seq = I2_seq + I1_seq;                                                    % summing holograms
    clear I1_seq
end
%%% Computing Average Hologram
I_avgseq = I2_seq/(HoloNum+1);
%%
%%% Generate 10 random numbers between 3001 and last hologram
% rng default
rnd = 1 + round(randperm(HoloNum,nums));
%%% Initialization step
Time = 0;
pp = 1;
mask = zeros(N,M);
[u,v] = meshgrid(1:N,1:M);
mask(((Lambda.*(u-(N/2+1))./Area).^2+(Lambda.*(v-(N/2+1))./Area).^2)<=1)=1;
I_recstack = zeros(N,M,length(dr1:depth:dr2));
I_recstack1 = zeros(N,M,length(dr1:depth:dr2));
SS =  zeros(N,M,length(dr1:depth:dr2));
AA(1,1,:)=(-2*pi*1i.*(dr1:depth:dr2)./Lambda);
S=@(u,v) exp(AA.*((sqrt(1-(Lambda.*(u-(N/2+1))./Area).^2-(Lambda.*(v-(N/2+1))./Area).^2).*ones(1,1,length(dr1:depth:dr2)))));
SS=S(u,v);
SS(mask==0)=0;
Data_res = zeros(100,1302);       %initiializing final results

% ind = [85,86,88,90,92,93,96,99,100];
for H = 6001:13000
    tic;
    H
%     H = Holo(t)
    %%% Loading raw hologram
    filename = ['MCF7_C_10_20X_420fps_texp_35us_Z_exp_200um_' num2str(H,'%05d')];
    I_raw = double(imread([filename '.tif']));
%     figure;imshow(I_raw,[],'InitialMagnification','fit')
    % Computing cleaned hologram by subtracting object free average from
    % raw hologram
    I_clean = I_raw - I_avgseq;
    I_clean1 = I_clean + abs(min(I_clean(:)));
%     figure;imshow(I_clean1,[],'InitialMagnification','fit')
    I_recstack = abs(ifft2(fftshift(fft2(I_clean)).*SS));
    I_recstack1 = abs(ifft2(fftshift(fft2(I_clean1)).*SS));
    I_recproj = max(I_recstack,[],3);
    I_recproj1 = max(I_recstack1,[],3);
%     figure;imshow(I_recproj1,[],'InitialMagnification','fit')
    % THRESHOLDING-1 (THRESHOLD DETERMINATION AND SEGMENTATION OF PROJECTION 2D IMAGE)
    D3_1 = mat2gray(I_recproj);
    T = adaptthresh(D3_1,'Statistic','gaussian','NeighborhoodSize',sz_windthresh);
    D3_2 = imbinarize(D3_1,k_thresh.*T);
    sel1 = strel(ones(el1_seg,el1_seg));
    sel2 = strel(ones(el2_seg,el2_seg));
    D3_5 = imerode(D3_2,sel1);
    D3_6 = imdilate(D3_5,sel2);
    D3_6 = imfill(D3_6,'holes');
    L1 = D3_6;
    %%%  COORDINATES AND SIZE DETERMINATION OF PARTICLE IMAGES IN 2D PROJECTION IMAGES
    E1 = L1;
    if max(E1(:) == 1)
        mets = regionprops(logical(E1),I_recproj,'WeightedCentroid');
        x_r = zeros(length(mets),1);
        y_r = zeros(length(mets),1);
        Int_max = zeros(length(mets),1);
        for i=1:length(mets)
            x_r(i) = round(mets(i).WeightedCentroid(1));
            y_r(i) = round(mets(i).WeightedCentroid(2));
        end
        Cd1 = [x_r y_r];
        [s1,t1] = find(x_r > nr & x_r < N - nr);
        x_r1 = x_r(s1);
        y_r1 = y_r(s1);
        [s2,t2] = find(y_r1 > nr_y & y_r1 < N - nr_y);
        x_r2 = x_r1(s2);
        y_r2 = y_r1(s2);
        Cd2 = [y_r2 x_r2];
        %%% DETERMINATION OF Z-LOCATION OF PARTICLE IMAGE IN 3D RECONSTRUCTION VOLUME
        s1 = size(Cd2,1);
        I13 = zeros(p,s1);
        Iz_pix = zeros(p,s1);
        %%% Laplacian-based axial Intensity profile
        m = zeros(s1,1);
        n = zeros(s1,1);
        for k1 = 1:p
            I11 = I_recstack(:,:,k1);
            for q = 1:s1
                m1 = Cd2(q,1);
                n1 = Cd2(q,2);
                I12(m1-R:m1+R,n1-R:n1+R) = I_recproj(m1-R:m1+R,n1-R:n1+R);    % create 7x7 pixel area mask around detected centroid
                [m(q), n(q)] = find(I12 == max(I12(:)));                       % Find max pixel value
                m(q) = (m(q))';n(q) = (n(q))';
                clear I12
                Gmag = 1;
                if (m(q)> nr || m(q) < N - nr) || (n(q)> nr || n(q) < N - nr)
                    for k2 = m(q)-1:m(q)+1
                        for k3 = n(q)-1:n(q)+1
                            Gmag = Gmag * (abs(2*I11(k2,k3)-I11(k2-1,k3)-I11(k2+1,k3))+abs(2*I11(k2,k3)-I11(k2,k3-1)-I11(k2,k3+1)))^2;
                        end
                    end
                end
                I13(k1,q) = Gmag;
                Iz_pix(k1,q) = I11(m(q),n(q));
            end
        end
        Cd3 = [m, n];
        %%% Normalized axial intensity profile
        Max = max(I13,[],1);                % this will have dimension: [1,# of particles]
        Min = min(I13,[],1);
        I14 = (I13 - repmat(Min,p,1))./(repmat(Max,p,1) - repmat(Min,p,1));
        %%% Z-location of centroid
        [i, j] = find(I14 == 1);        % i: z-plane, j:particle index
        Z_Loc1 = i;                     % this will have dimension: [# of particles,1]
        %%% Storing (x,y,z) coordinates of detected centroid
        Cd4(:,1) = Cd3(:,1);
        Cd4(:,2) = Cd3(:,2);
        Cd4(:,3) = Z_Loc1;
        Cd4(:,4) = di+(Z_Loc1.*depth);
        %%%%%%%%%%%%%%%%%%%%%%%%%Fingerprinting%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        s1 = size(Cd4, 1);
        for p1 = 1:s1
%             if (Cd4(p1,4)>= di+190 & Cd4(p1,4) <= di+490)
            m1 = Cd4(p1,1);
            n1 = Cd4(p1,2);
            z1 = Cd4(p1,3);
            z2 = di+(z1*depth);
            Iz_max = max(I13(:,p1),[],1);
            Iz_pixmax = max(Iz_pix(:,p1),[],1);
            if Iz_pixmax > 35
                I_pbf = I_recstack1(:,:,z1);
                cell = imcrop(uint8(I_pbf),[n1-floor(W11/2) m1-floor(W11/2) W11 W11]);
%                 figure;imshow(cell,[])
                    Data_res(pp,1) = H;                     % records frame of detection
                    Data_res(pp,2) = m1;                    % 'x' is in um
                    Data_res(pp,3) = n1;                    % 'y' is in um
                    Data_res(pp,4) = z1;                    %  z-plane
                    Data_res(pp,5) = z2;                    %  z in um
                    Data_res(pp,6) = Iz_pixmax;             % maximum single pixel axial intesnity
                    Data_res(pp,7:1302) = cell(:);          % unwrap 36x36 plane off best focus image for each detected cell
                    pp = pp+1;
                    clear I_pbf cell
            end
%             end
        end
        toc;
        clear Cd1 Cd2 Cd3 Cd4
    end
    Time = Time + toc;
%     figure;
%     imshow(I_clean1,[],'InitialMagnification','fit');
%     hold on
%     plot(Data_res(:,3),Data_res(:,2),'r+','MarkerSize',5)
%     filename = ['Detections_' num2str(H, '%05d')];
%     dest_folder = 'C:\Users\Anivader\OneDrive - Texas Tech University\Ani\Projects\Labyrinth DHM sheath spiked cell\Videos\PureWBC_100k\';
%     imwrite(uint8(I_clean1),strcat(dest_folder,'CleanedHologram_pureWBC','_',num2str(H),'.tif'));
%     saveas(f,strcat(dest_folder,'Spiked_C_10','_',num2str(H),'.jpg'));
%     Data_res = zeros(100,1302);
end
%%%
%%% Removing rows containing "0's"
[i,j] = find(Data_res(:,1) ~= 0);
Data_res = Data_res(i,:);
filename = '20220818_MCF7_spiked_C_10_H_6001_to_13000.mat';
dest_folder = 'H:\Processed holograms_dotmat files\';
save(strcat(dest_folder,filename),'Data_res')
%
%%
%%% Figures
% figure;
% imagesc(I_raw)
% colormap(gray)
% figure;
% imagesc(I_avgseq)
% colormap(gray)
% figure;
% imagesc(I_clean1)
% colormap(gray)
% figure;
% imagesc(I_recproj)
% colormap(gray)
% figure;
% imagesc(I_recproj1)
% colormap(gray)
% figure;
% imagesc(I_clean1)
% colormap(gray)
% hold on
% plot(Data_res(:,3),Data_res(:,2),'r+','MarkerSize',5,'LineWidth',1)
% figure;
% imagesc(I_clean1)
% colormap(gray)
% hold on
% plot(xx(t),yy(t),'r+','MarkerSize',5,'LineWidth',1)
% figure;
% imagesc(I_recproj)
% colormap(gray)
% hold on
% plot(Data_res(:,3),Data_res(:,2),'r+','MarkerSize',5)
% figure;
% imagesc(I_recproj1)
% colormap(gray)
% hold on
% plot(Data_res(:,3),Data_res(:,2),'r+','MarkerSize',5,'LineWidth',1)
% figure;
% imagesc(Io1)
% colormap(gray)
% hold on
% plot(Cd4(:,2),Cd4(:,1),'r+','MarkerSize',5)
% % % figure;imshow(Io1,[],'InitialMagnification','fit')
% figure;imshow(L1,'InitialMagnification','fit')
% % % hold on
% % % plot(Data_res(:,3),Data_res(:,2),'r+','MarkerSize',5)
% %
%%%
% figure;
% scatter(round(Data_res(:,5)),Data_res(:,2).*dx,50,'MarkerEdgeColor','k',...
%     'MarkerFaceColor',[0 1 0.75])
% hold on
% xf = di+190;
% h = 330;
% x1 = xf.*ones(1000,1);
% x2 =  (xf+h) .*ones(1000,1);
% y = linspace(0,N*dx,1000);
% plot(x1,y,'r--',x2,y,'r--')
% plot(x1,y,'r--')
% set(gca,'FontSize',16,'FontWeight', 'bold')
% xt = get(gca, 'XTick');
% axis([-50 1000  0 N.*dx])
% axis square
% ax = gca;
% ax.LineWidth = 2;
% box on
% xlabel('z in \mum')
% ylabel('y in \mum')
% xticks(-100:100:1000)
% yticks(0:200:N*dx)
% title('M = 20X','fontsize',20)
% title('M=20X, Z_{rec} = 300\mum, t_{exp} = 20\mus','fontsize',20)
% figure;imshow(Io1,[],'InitialMagnification','fit')

%%
%%%%%%%%%%%%%%%%%%%%%%%%% Removing multuiple counts%%%%%%%%%%%%%%%%%%%%%%%%
%%% Removing detections in sheath layer
yy = Data_res(:,2);
i = find(yy >= 125 & yy <= N - 125); 
Data_res1 = Data_res(i,:);
%
%%% Removing background objects/debris
Iz_pixmax = Data_res1(:,6);
thresh = 35;
j = find(Iz_pixmax > thresh);
Data_res2 = Data_res1(j,:);
%%
%%% Applying MCR algorithm
L1_T = Data_res2;
L2_T = Data_res2;
s3 = size(L1_T,1);
disp('Removing multiple counts ...')
for k1 = 1:s3
    H = L1_T(k1,1);                                      % hologram frame
    y1 = L1_T(k1,2);                                     % y coordinate
    z1 = L1_T(k1,4);
    if L2_T(k1,3) ~= 0
        for k2 = 1:s3
            if (L1_T(k2,1) >= H+1 && L1_T(k2,1) <= H+4) && (L1_T(k2,2) >= y1-3 && L1_T(k2,2) <= y1+3) && (L1_T(k2,4) >= z1-10 && L1_T(k2,4) <= z1+10)
                if L1_T(k2,3) > L1_T(k1,3)
                    L2_T(k2,3) = 0;
                end
            end
        end
    end
end
s4 = size(L2_T,1);
a = 1;
for k1 = 1:s4
    if L2_T(k1,3) ~= 0
        L3_T(a,:) = L2_T(k1,:);
        a = a+1;
    end
end
%%
%%% Saving MCR results
filename = '20220818_MCF7_spiked_C_10_H_6001_to_13000_MCR.mat';
dest_folder = 'H:\Processed holograms_dotmat files\';
save(strcat(dest_folder,filename),'L3_T')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%Generate image library%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L1 = L3_T(:,7:end);
% Choose selected area around cell center 
N = 36;
crop_sz = 36;
L1_crop_lin = zeros(length(L1),crop_sz^2);
disp('Generating library of cropped images ...')
for s = 1:length(L1)
    L1_mat = reshape(L1(s,:),N,N);
%     figure;imshow(L1_mat,[])
    dest_folder = 'H:\Experiments\Spiked cell DHM-sheath_ML\Pdhm setup\20220818_Labyrinth_ML_3cycles\Images for Network\C_10_H_6001_to_13000\';
    imwrite(uint8(L1_mat),strcat(dest_folder,'Mixed','_',num2str(s),'.tif'));
    clear L1_mat
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Generate ML prediction%%%%%%%%%%%%%%%%%%%%%
clear; 
clc;

%%% Loading images
%%% Select folder that contains cropped MCF7 imageset you wish to use for
%%% training
imgs_path = uigetdir('H:\Experiments\Spiked cell DHM-sheath_ML\Pdhm setup\20220818_Labyrinth_ML_3cycles\Images for Network\C_10_H_6001_to_13000\');                    % Select folder where object holograms are stored
%
cd(imgs_path)
cd ..
[~,name] = fileparts(imgs_path);
%
%%% Creating image datastore
imds = imageDatastore(name,'IncludeSubfolders',true,'LabelSource','foldernames');
T = countEachLabel(imds);
imgTotal = length(imds.Files);
%
%%% Displaying some a few cell images
% figure;
% numImages = imgTotal/2;
% num = 36;
% rnd1 = randperm(numImages,num);
% for i = 1:num
%     subplot(6,6,i);
%     imshow(imds.Files{rnd1(i)});
%     drawnow;
% end
%
%%% Assign images to testing set
imds_Test = imds;
%
%%% Loading saved model
Model_path = uigetdir('H:\Trained DL models\CNN_repeatability\Ntrain_36638_Ntest_15702\');
cd(Model_path)
[filename, path] = uigetfile('*.mat');
load(filename,'net')
%
%%% Get predictions on test set
disp('Generating predictions ...')
tic
predictTestLabels = classify(net,imds_Test);
toc
%
%%% Get Softmax probabilities
disp('Generating Softmax probabilities ...')
act_prob = activations(net,imds_Test,'softmax','OutputAs','rows');
disp('Generated Softmax probabilities ...')
%
%%% Apply threshold to reduce FPs
%%% Set threshold on probability
format long
dec_thresh = 0.9999999;
%%% Find MCF7s predicted based on decision criterion
k2 = find(act_prob(:,1)>dec_thresh);
n_target = length(k2);
V_analyzed = 2/3;
C_target = n_target/V_analyzed;

disp(['Cancer cell count predicted by ML:   ' num2str(n_target) ''])
disp(['Cancer cell load predicted by ML:   ' num2str(C_target) ' per mL'])

%%
%%% Show target cell detections
% figure;
% for m = 1:length(k2)
%     subplot(10,10,m)
%     mm = k2(m);
%     img1 = readimage(imds_Test,mm);
%     imshow(img1,[])
%     drawnow
%     clear img1
% end










