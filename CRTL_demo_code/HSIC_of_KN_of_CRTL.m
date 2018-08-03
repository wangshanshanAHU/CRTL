function   [Z,P,Ks, Kt,KY_test]= HSIC_of_KN_of_CRTL(Xt_test,X,Xs_train,Xt_train,Xs_label,Xt_label,alphaG,alphaJ,alphaZ,alphaZ3,alphaZ2,Kernel) 
% The main function of Supervised Regularization based Robust Subspace (SRRS) Method.

% Author: Sheng Li (shengli@ece.neu.edu)
% June 29, 2015
C= length(unique(Xs_label));
m=size(Xs_train,1);
Ns=size(Xs_train,2);
Nt=size(Xt_train,2);
Ntu=size(Xt_test,2);
NK=size(X,2);
H=eye(Nt+Ns,Nt+Ns)-ones(Nt+Ns,Nt+Ns)/(Nt+Ns);
dim=NK;
 switch Kernel 
       case'linear'    
       kervar1=1.2;% free parameter
       kervar2=10;% no use 
       case  'gauss'
       kervar1=1.2;% free parameter
       kervar2=10;% no use
 end    
       X = X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);  %//归一化
       K    = gram(X',X',Kernel,kervar1,kervar2);
       K =max(K,K');
       Kt   = gram(X',Xt_train',Kernel,kervar1,kervar2);
       Ks   = gram(X',Xs_train',Kernel,kervar1,kervar2);
       KY_test  = gram(X',Xt_test',Kernel,kervar1,kervar2);
   
[A Beta]    = eig(K) ;
[dummy id]  = sort(diag(Beta),'descend');
A           = A(:,id(1:dim)) ;A=real(A);
P1=A;

X_train_label=[Xs_label;Xt_label];
Y = Construct_Y(X_train_label,length(X_train_label)); 
L= gram(Y',Y',Kernel,kervar1,kervar2);

%%% Parameter setting in ALM

Bc=cell(1,C);
A=cell(C,1);
Xsc=cell(C,1);
Xtc=cell(C,1);

Ksc=cell(C,1);%%%核函数
Ktc=cell(C,1);%%%核函数

Ac1=zeros(Nt);
Bc1=zeros(Ns);

%%% Parameter setting in ALM_Z
rho = 1.01;
max_mu = 1e6;
mu = 0.01;
maxIter = 3e2;
convergence = 10^-6;

R2 = zeros(Ns,Nt);
R3 = zeros(Ns,Nt);
R6 = zeros(1,Nt);
G=zeros(Ns,Nt);
J=zeros(Ns,Nt);

 for c=1:C 
     Ntc= length(find(Xt_label==c));
     Nsc= length(find(Xs_label==c));  
     Xtc_train= Xt_train(:,find(Xt_label==c));
     Xsc_train= Xs_train(:,find(Xs_label==c));
     Ktc_train = gram(X',Xtc_train',Kernel,kervar1,kervar2);%%%核函数
     Ksc_train = gram(X',Xsc_train',Kernel,kervar1,kervar2);%%%核函数
       Ktc{c}=Ktc_train;%%%核函数
       Ksc{c}=Ksc_train;%%%核函数
     c_col= find(Xt_label==c);
       for j=1:Ntc
            Ac1(c_col(j,1),j)=1;
       end
      Ac2=eye(Nt,Ntc); 
      A{c}=Ac1*Ac2;  
      Ac1=zeros(Nt);           
      c_row= find(Xs_label==c);                      
       for j=1:Nsc
            Bc1(j,c_row(j,1))=1;
       end
      Bc2=eye(Nsc,Ns);
      Bc{c}=Bc2*Bc1;
      Bc1=zeros(Ns);
 end

 P_dis=K*H*L*H*K/((Nt+Ns-1)*(Nt+Ns-1));
alphaZ2=1/((Nt+Ns)*(Nt+Ns));  %B1
alphaZ3=1/((Nt+Ns)*(Nt+Ns));  %B2

%% The main loop of optimization
iter = 0;
while iter<3
    
K_Ph2=0;
K_Ph3=0;

K_Zh3=0;
K_Zh2=0;

    iter = iter + 1;                         
    fprintf('iter =%d ',iter);
      if iter == 1
          Z =ones(Ns,Nt); 
           P=P1;
      end
      
 %%% Update P     
        for c=1:C             
         for k=1:C
             if k==c
                Ntc= length(find(Xt_label==c));
                K_H3ck=zeros(NK,Ntc);%%%核函数
             else
                K_H3ck=Ksc{c}*Bc{c}*Z*A{k};%%%核函数
             end
           K_Ph_3=alphaZ3*(K_H3ck)*(K_H3ck)';%%%核函数
           K_Ph3=K_Ph3+K_Ph_3;%%%核函数
         end        
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%核函数
         K_H2c=Ktc{c}-(Ksc{c}*Bc{c}*Z*A{c});
         K_Ph_2=alphaZ2*(K_H2c)*(K_H2c)';
         K_Ph2=K_Ph2+K_Ph_2;
       end  
          K_Ph=-(K_Ph2+K_Ph3+P_dis);
        P= UpdateP(K_Ph,K); %%特征值分解方法

    %%% Update Z     
   for itemZ=1:25      
       for  c=1:C         
           for k=1:C
               if k==c
                  K_H3z=zeros(Ns,Nt);
               else
                  K_H3z=(Bc{c})'*(Ksc{c})'*P*P'*(Ksc{c})*Bc{c}*Z*A{k}*A{k}';%%%核函数
                  K_H3z=alphaZ3*K_H3z;%%%核函数
               end       
                 K_Zh3=K_Zh3+K_H3z;%%%核函数
           end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%核函数 
               K_H2z=-alphaZ2*(Bc{c})'*(Ksc{c})'*P*(P'*Ktc{c}-P'*Ksc{c}*Bc{c}*Z*A{c})*(A{c})';
              K_Zh2=K_Zh2+K_H2z;    
       end 
%        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          K_Zh6=mu*ones(Ns,1)*(ones(1,Ns)*Z-ones(1,Nt));
          K_Derta_Zold=K_Zh2+K_Zh3+R2/2+R3/2+(ones(Ns,1)*R6/2)+(mu*(Z-J))+(mu*(Z-G))+K_Zh6; %Z2=Z*Z2 
        Derta= K_Derta_Zold/norm(K_Derta_Zold,2);             
          Z_iter=alphaZ*Derta;
          Z=Z-Z_iter; 
           
   %%% Update J
    temp = Z + R2/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>alphaJ/mu));
    if svp>=1
        sigma = sigma(1:svp)-alphaJ/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    
    %%% Update G     
    taa =alphaG /mu;
    tempG =  Z + R3/mu;
    G = max(0,tempG-taa)+min(0,tempG+taa);
    
     
    % updating R2, R3, R4
    R2 = R2+(mu*(Z-J));
    R3 = R3+(mu*(Z-G));
    R6 = R6+(mu*(ones(1,Ns)*Z-ones(1,Nt)));
    
    % updating mu
    mu = min(rho*mu,max_mu);
  
   end   %%%%%跟Z迭代对应        
end

end

function [P] = UpdateP(Ph,K)

ReducedDim = size(Ph,2); 
   A=pinv(K)*Ph; 
    A_eig=(A+A')/2;
 [eigV, eigD] = eig(A_eig);
[~, d_site] = sort(diag(eigD),'ascend');
V = eigV(:,d_site(1:ReducedDim));
P = V;
end

function Y = Construct_Y(gnd,num_l)
%%
% gnd:标签向量；
% num_l:表示有标签样本的数目；
% Y:生成的标签矩阵；
nClass = length(unique(gnd));
Y = zeros(nClass,length(gnd));
for i = 1:num_l
    for j = 1:nClass
        if j == gnd(i)
            Y(j,i) = 1;
        end  
    end
end
end


   
           
   
           

 




