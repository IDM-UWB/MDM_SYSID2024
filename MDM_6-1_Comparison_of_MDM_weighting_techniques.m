clc
close all
clear all

% MDM, LTV, Example TAC

rng(0) % R2019b

Number = 1e3; % Number of measurements
MC = 1e4; % Number of MC simulations

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% System / Model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 1
nx = 1;
F = cell(Number,1);
G = cell(Number,1);
E = cell(Number,1);
u = cell(Number,1);

H = cell(Number,1);
D = cell(Number,1);
nz = nan(Number,1);

nw = 1;
nv = 1;
Q = 2;
R = 1;
for t=1:Number
    F{t} = 0.8-0.1*sin(7*pi*t/Number);    
    G{t} = 1;   
    u{t} = sin(t/Number);   
    E{t} = 1; 
     
    nz(t) = 1;
    H{t} = 1+0.99*sin(100*pi*t/Number);
    D{t} = 1;   
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%% End: System / Model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
startOld=now;
for iMC=1:MC
start=now;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data generator %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 1
w = chol(Q)'*randn(nw,Number);
v = chol(R)'*randn(nv,Number); 
x = nan(nx,Number);
z = cell(Number,1);
x(:,1) = 1 + rand(nx,1);
for t=1:Number
    if t<Number
        x(:,t+1) = F{t} * x(:,t)  + G{t} * u{t} + E{t} * w(:,t); 
    end
    z{t} = H{t} * x(:,t) + D{t} * v(:,t);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%% End: Data generator %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%% MDM matrices and vectors %%%%%%%%%%%%%%%%%%%%%%%%%
if 1
L = 1;
N = 1;
if iMC==1 
    [A2u,covRes,Mat_covRes,Xi_A2] = MDM(L,N,F,G,E,nz,H,D,z,u);
else
    covRes = MDM_covRes(L,N,G,z,u,Mat_covRes);
end
end
%%%%%%%%%%%%%%%%%%%%% End: MDM matrices and vectors %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%% MDM - unweighted, non-recursive %%%%%%%%%%%%%%%%%%%%%%
if 1
tic
A2u_UwNr = vertcat(A2u{:});
QR_UwNr_cov(:,:,iMC) = (A2u_UwNr'*A2u_UwNr)\eye(size(A2u_UwNr,2));
QR_UwNr(:,iMC) = QR_UwNr_cov(:,:,iMC)*A2u_UwNr'*vertcat(covRes{:});
time_UwNr(iMC) = toc;
end
%%%%%%%%%%%%%%%%%% End: MDM - unweighted, non-recursive %%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%% MDM - semi-weighted, non-recursive %%%%%%%%%%%%%%%%%%%%%
if 1
tic
%%%%%% Matrix for weighting %%%%%
inv_cov_covRes = cell(Number-L+1,1);
for t=1+N:Number-L+1
    cov_covRes_t = Xi_A2{t} * Xi_A2{t}';
    inv_cov_covRes{t} = cov_covRes_t\eye(size(cov_covRes_t,1)); % inv(cov_covRes_t)
end
inv_cov_covRes_diag = blkdiag(inv_cov_covRes{:});
%%% End: Matrix for weighting %%%

A2u_SwNr = vertcat(A2u{:});

QR_SwNr_cov(:,:,iMC) = (A2u_SwNr'*inv_cov_covRes_diag*A2u_SwNr)\eye(size(A2u_SwNr,2));
QR_SwNr(:,iMC) = QR_SwNr_cov(:,:,iMC)*A2u_SwNr'*inv_cov_covRes_diag*vertcat(covRes{:});
time_SwNr(iMC) = toc;
end
%%%%%%%%%%%%%%% End: MDM - semi-weighted, non-recursive %%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%% MDM - weighted(UwNr), non-recursive %%%%%%%%%%%%%%%%%%%%
if 1
%%%%%% Matrix for weighting %%%%%
if iMC==1
    for timeShift = 0:L+N-1 
        [EwvLS4_fun{timeShift+1},QRu_sim] = EwvLS4_compute_Fast(L+N,timeShift,nw,nv);
    end
end
%%% End: Matrix for weighting %%%
tic
%%% 4. moments - UwNr estimate is used for weight matrix %%%
nMNumber = size(1+N:Number-L+1,2);
EwvLS4_all = 0;
for timeShift = 0:L+N-1
    EwvLS4_UwNr = double(subs(EwvLS4_fun{timeShift+1},QRu_sim,QR_UwNr(:,iMC)));
    EwvLS4_part = kron([zeros(timeShift,nMNumber);eye(nMNumber-timeShift,nMNumber)], EwvLS4_UwNr);
    EwvLS4_all = EwvLS4_all + EwvLS4_part;
    if timeShift>0
        EwvLS4_all = EwvLS4_all + EwvLS4_part';
    end 
end
%%% 4. moments - UwNr estimate is used for weight matrix %%%

blkdiag_Xi_A2 = blkdiag(Xi_A2{:});
cov_covRes = blkdiag_Xi_A2 * EwvLS4_all * blkdiag_Xi_A2';
inv_cov_covRes = cov_covRes\eye(size(cov_covRes,1));

A2u_WeNr = vertcat(A2u{:});

QR_WeNr_cov(:,:,iMC) = (A2u_WeNr'*inv_cov_covRes*A2u_WeNr)\eye(size(A2u_WeNr,2));
QR_WeNr(:,iMC) = QR_WeNr_cov(:,:,iMC)*A2u_WeNr'*inv_cov_covRes*vertcat(covRes{:});
time_WeNr(iMC) = toc;
end
%%%%%%%%%%%%%%%% End: MDM - weighted(UwNr), non-recursive %%%%%%%%%%%%%%%%%

if mod(iMC,10)==0; disp([' MDM estimate - ', num2str(iMC/(MC)*100),'% ']); end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 1
mean_QR_UwNr = mean(QR_UwNr,2);
cov_QR_UwNr = cov(QR_UwNr');
est_QR_UwNr_cov = mean(QR_UwNr_cov,3);

mean_QR_SwNr = mean(QR_SwNr,2);
cov_QR_SwNr = cov(QR_SwNr');
est_QR_SwNr_cov = mean(QR_SwNr_cov,3);

mean_QR_WeNr = mean(QR_WeNr,2);
cov_QR_WeNr = cov(QR_WeNr');
est_QR_WeNr_cov = mean(QR_WeNr_cov,3);

[[Q;R],mean_QR_UwNr,mean_QR_SwNr,mean_QR_WeNr]

[diag(cov_QR_UwNr),diag(cov_QR_SwNr),diag(cov_QR_WeNr)]

[diag(est_QR_UwNr_cov),diag(est_QR_SwNr_cov),diag(est_QR_WeNr_cov)]

[norm(cov_QR_UwNr-est_QR_UwNr_cov)/norm(est_QR_UwNr_cov),...
 norm(cov_QR_SwNr-est_QR_SwNr_cov)/norm(est_QR_SwNr_cov),...
 norm(cov_QR_WeNr-est_QR_WeNr_cov)/norm(est_QR_WeNr_cov)]

[mean(time_UwNr), mean(time_SwNr), mean(time_WeNr);...
 mean(time_UwNr)/mean(time_UwNr), mean(time_SwNr)/mean(time_UwNr), mean(time_WeNr)/mean(time_UwNr)]
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End: Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 1
figure(1)
hold all
grid on
xlim([1.75 2.25])
ylim([0.8 1.2])
xlabel('Q','Interpreter','tex')
ylabel('R','Interpreter','tex')

circle = [sin(linspace(0,2*pi,80));cos(linspace(0,2*pi,80))];

p0 = plot(Q,R,'ko','MarkerSize',40,'LineWidth',0.5);

p1 = plot(mean_QR_UwNr(1),mean_QR_UwNr(2),'b+','MarkerSize',30);
elips_QR_UwNr = mean_QR_UwNr + chol(cov_QR_UwNr)'*circle;
p2 = plot(elips_QR_UwNr(1,:),elips_QR_UwNr(2,:),'b-','LineWidth',1.2);
est_elips_QR_UwNr = mean_QR_UwNr + chol(est_QR_UwNr_cov)'*circle;
p3 = plot(est_elips_QR_UwNr(1,:),est_elips_QR_UwNr(2,:),'b--','LineWidth',1.2);

p4 = plot(mean_QR_SwNr(1),mean_QR_SwNr(2),'rsquare','MarkerSize',30);
elips_QR_SwNr = mean_QR_SwNr + chol(cov_QR_SwNr)'*circle;
p5 = plot(elips_QR_SwNr(1,:),elips_QR_SwNr(2,:),'r-','LineWidth',1.2);
est_elips_QR_SwNr = mean_QR_SwNr + chol(est_QR_SwNr_cov)'*circle;
p6 = plot(est_elips_QR_SwNr(1,:),est_elips_QR_SwNr(2,:),'r--','LineWidth',1.2);

p7 = plot(mean_QR_WeNr(1),mean_QR_WeNr(2),'x','color',[0 .5 .2],'MarkerSize',30);
elips_QR_WeNr = mean_QR_WeNr + chol(cov_QR_WeNr)'*circle;
p8 = plot(elips_QR_WeNr(1,:),elips_QR_WeNr(2,:),'-','color',[0 .5 .2],'LineWidth',1.2);
est_elips_QR_WeNr = mean_QR_WeNr + chol(est_QR_WeNr_cov)'*circle;
p9 = plot(est_elips_QR_WeNr(1,:),est_elips_QR_WeNr(2,:),'--','color',[0 1 1],'LineWidth',1.2);

pn = plot(0,nan,'w.');

legend([p0,p1,p4,p7,pn,p2,p5,p8,pn,p3,p6,p9],{'True','UwNr - MC sample mean', 'SwNr - MC sample mean', 'WeNr - MC sample mean',...
                                        '    ','UwNr - MC sample std', 'SwNr - MC sample std', 'WeNr - MC sample std',...
                                        '    ','UwNr - average est std', 'SwNr - average est std', 'WeNr - average est std'},'NumColumns',3)
%%% Estimates
%plot(QR_UwNr(1,:),QR_UwNr(2,:),'r+','MarkerSize',1);
%plot(QR_SwNr(1,:),QR_SwNr(2,:),'rx','MarkerSize',1);
%plot(QR_WeNr(1,:),QR_WeNr(2,:),'ro','MarkerSize',1);
%%%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End: Plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%save('Example_TAC.mat')