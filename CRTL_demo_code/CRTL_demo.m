clear all
close all
addpath libsvm-new
%%%%%%%%%%高斯核参数1.2
src_strname = {'amazon','Caltech10','webcam','amazon','webcam','dslr','dslr','webcam','Caltech10','Caltech10','dslr','amazon'};
tgt_strname = {'dslr','dslr','dslr','Caltech10','Caltech10','Caltech10','amazon','amazon','amazon','webcam','webcam','webcam'};
load('data\optimal_parameters_svm.mat');
Data=load('4DA-CNN\DeCAF_Data.mat');
src_str = {Data.amazon_data,Data.caltech_data,Data.webcam_data,Data.amazon_data,Data.webcam_data,Data.dslr_data,Data.dslr_data,Data.webcam_data,Data.caltech_data,Data.caltech_data,Data.dslr_data,Data.amazon_data};
slabel_str={Data.amazon_label,Data.caltech_label,Data.webcam_label,Data.amazon_label,Data.webcam_label,Data.dslr_label,Data.dslr_label,Data.webcam_label,Data.caltech_label,Data.caltech_label,Data.dslr_label,Data.amazon_label};
tgt_str={Data.dslr_data,Data.dslr_data,Data.dslr_data,Data.caltech_data,Data.caltech_data,Data.caltech_data,Data.amazon_data,Data.amazon_data,Data.amazon_data,Data.webcam_data,Data.webcam_data,Data.webcam_data};
tlabel_str={Data.dslr_label,Data.dslr_label,Data.dslr_label,Data.caltech_label,Data.caltech_label,Data.caltech_label,Data.amazon_label,Data.amazon_label,Data.amazon_label,Data.webcam_label,Data.webcam_label,Data.webcam_label};
for i_sam = 1:length(src_strname)

    alphaG=1*alpha(i_sam);
    alphaJ=1*beta(i_sam);
    alphaZ=lambda(i_sam);
    alphaZ3=gama3(i_sam);
    alphaZ2=gama2(i_sam);
    %----------random-------------------------------   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    src = src_str{i_sam};
    tgt = tgt_str{i_sam};

    fts=src{2};
    labels=slabel_str{i_sam};

    Xs = fts';
    Xs_label = labels;
    clear fts;
    clear labels;

    fts=tgt{2};
    labels=tlabel_str{i_sam};
    Xt = fts';
    Xt_label = labels;
    clear fts;
    clear labels;
    
    src = src_strname{i_sam};
    tgt = tgt_strname{i_sam};
    fprintf(' %s vs %s ', src, tgt);
   load(strcat('data\SameCategory_',src, '-',tgt, '_20RandomTrials_10Categories.mat')); 
    %----------random-------------------------------
     if strcmp(src,'amazon_data')
        train_num=20;
     else
        train_num=8;
     end
     
    Xs_r = Xs./repmat(sqrt(sum(Xs.^2)),[size(Xs,1) 1]); 
    Xt_r = Xt./repmat(sqrt(sum(Xt.^2)),[size(Xt,1) 1]);     
     for i=1:20
    Xs_train=Xs_r(:,train.source{i});source_label_train= Xs_label(train.source{i});
    Xt_train=Xt_r(:,train.target{i});target_label_train=Xt_label(train.target{i});
    Xt_test=Xt_r(:,test.target{i});target_label_test=Xt_label(test.target{i});
    %_______________________________________________
    % randomly select 8 samples per class from source domain (webcam and dslr data)
    % ------------------------------------------
    %             Transfer Learning
    % ------------------------------------------ 
   ker_type=1; % 0:linear 1:nonlinear
   if ker_type==0
      Kernel='linear';
       X=[Xs_train,Xt_train];  
   else
     Kernel='gauss'; 
      X=[Xs_train,Xt_train];  
   end
   [Z,P,Ks, Kt,KY_test]= HSIC_of_KN_of_CRTL(Xt_test,X,Xs_train,Xt_train,source_label_train,target_label_train,alphaG,alphaJ,alphaZ,alphaZ3,alphaZ2,Kernel);   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     KX_train  = Ks;
     KY_train  = Kt;
     KY_test  = KY_test;
     
   X_train  = P'*KX_train;
   Y_train  = P'*KY_train;
   Xt_test_new  = P'*KY_test; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % -------------------------------------------
    %               Classification
    % ------------------------------------------- 
   Class= length(unique(source_label_train)); 
  % ls
  Yat=[source_label_train;target_label_train];%target_label_train
  Xat=[X_train(:,1:length(source_label_train)),Y_train(:,1:length(target_label_train))]';  % Xt_new
   Y=-1*ones(length(Yat),Class); 
   for j=1:length(Yat)   
       Y(j,Yat(j))=1; 
   end  
%    search the best regularization parameter
   a=[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,100];
   for j=1:length(a)
       w=(Xat'*Xat+a(j)*eye(size(Xat,2)))\(Xat'*Y);
       yte1=Xt_test_new'*w;
       rate4_1(i,j)=decision_class(yte1,target_label_test); 
   end
   rate_ls(i)=max(rate4_1(i,:));
end            
ave_ls=mean(rate_ls);
fprintf('ave_ls= %2.2f%%\n',ave_ls);

end