clc
close all
clear all

% MDM, LTV

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
nv = 2;
Q=3;
R=[2 -1;-1 1];
for t=1:Number
    F{t} = 1 + 0.1*sin(20*pi*t/Number);              
    if t<Number/3
        nz(t) = 1;
        H{t} = 1;
        D{t} = [1 0];
    elseif t<2*Number/3       
        nz(t) = 1;
        H{t} = 1;
        D{t} = [0 1];  
    else
        nz(t) = 2;
        D{t} = [1 0;0 1];
        H{t} = [1;1];
    end
    E{t} = -1;
    G{t} = 1;
    u{t} = sin(t/Number); 
end
nQRu = nw*(nw+1)/2+nv*(nv+1)/2;
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
L = 2;
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

%%%%%%%%%%%%%%%%%%%%%% MDM - unweighted, recursive %%%%%%%%%%%%%%%%%%%%%%%%
if 1
tic
QR_UwRe_t = repmat(nan(nQRu,1),1,Number-L+1);
P_QR_UwRe_t = repmat(nan(nQRu),1,1,Number-L+1);
for t = 1+N:Number-L+1
    if t==1+N % Apriori estimate 
        QR_UwRe_apri = zeros(nQRu,1) + 0.5*[1;1;0;1];
        P_QR_UwRe_apri = 1e1*eye(nQRu);
    else
        QR_UwRe_apri = QR_UwRe_t(:,t-1);
        P_QR_UwRe_apri = P_QR_UwRe_t(:,:,t-1);
    end 
    cov_covRes_t = eye(size(A2u{t},1)); % Omega
    Gain = P_QR_UwRe_apri*A2u{t}'/(A2u{t}*P_QR_UwRe_apri*A2u{t}' + cov_covRes_t);
    QR_UwRe_t(:,t) = QR_UwRe_apri + Gain * (covRes{t} - A2u{t}*QR_UwRe_apri);
    P_QR_UwRe_t(:,:,t) = (eye(nQRu) - Gain * A2u{t}) * P_QR_UwRe_apri;
end
QR_UwRe_cov(:,:,:,iMC) = P_QR_UwRe_t;
QR_UwRe(:,:,iMC) = QR_UwRe_t;
time_UwRe(iMC) = toc;
end 
%%%%%%%%%%%%%%%%%%%%% End: MDM - unweighted, recursive %%%%%%%%%%%%%%%%%%%%

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
if iMC==1 % Rank check
    if rank(A2u_SwNr)<nQRu 
        rank(A2u_SwNr)
        error('The unique Q R parameters cannot be estimated')
    end
end % End: Rank check

QR_SwNr_cov(:,:,iMC) = (A2u_SwNr'*inv_cov_covRes_diag*A2u_SwNr)\eye(size(A2u_SwNr,2));
QR_SwNr(:,iMC) = QR_SwNr_cov(:,:,iMC)*A2u_SwNr'*inv_cov_covRes_diag*vertcat(covRes{:});
time_SwNr(iMC) = toc;
end
%%%%%%%%%%%%%%% End: MDM - semi-weighted, non-recursive %%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%% MDM - semi-weighted, recursive %%%%%%%%%%%%%%%%%%%%%%
if 1
tic
QR_SwRe_t = repmat(nan(nQRu,1),1,Number-L+1);
P_QR_SwRe_t = repmat(nan(nQRu),1,1,Number-L+1);
for t = 1+N:Number-L+1
    if t==1+N % Apriori estimate 
        QR_SwRe_apri = zeros(nQRu,1) + 0.5*[1;1;0;1];
        P_QR_SwRe_apri = 1e1*eye(nQRu);
    else
        QR_SwRe_apri = QR_SwRe_t(:,t-1);
        P_QR_SwRe_apri = P_QR_SwRe_t(:,:,t-1);
    end 
    cov_covRes_t = Xi_A2{t} * Xi_A2{t}'; % Omega 
    Gain = P_QR_SwRe_apri*A2u{t}'/(A2u{t}*P_QR_SwRe_apri*A2u{t}' + cov_covRes_t);
    QR_SwRe_t(:,t) = QR_SwRe_apri + Gain * (covRes{t} - A2u{t}*QR_SwRe_apri);
    P_QR_SwRe_t(:,:,t) = (eye(nQRu) - Gain * A2u{t}) * P_QR_SwRe_apri;
end
QR_SwRe_cov(:,:,:,iMC) = P_QR_SwRe_t;
QR_SwRe(:,:,iMC) = QR_SwRe_t;
time_SwRe(iMC) = toc;
end
%%%%%%%%%%%%%%%%% End: MDM - semi-weighted, recursive %%%%%%%%%%%%%%%%%%%%%

if mod(iMC,10)==0; disp([' MDM estimate - ', num2str(iMC/(MC)*100),'% ']); end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 1
mean_QR_UwNr = mean(QR_UwNr,2);
cov_QR_UwNr = cov(QR_UwNr');
est_QR_UwNr_cov = mean(QR_UwNr_cov,3);    
    
mean_QR_UwRe = mean(QR_UwRe(:,end,:),3);
cov_QR_UwRe = cov(squeeze(QR_UwRe(:,end,:))');
est_QR_UwRe_cov = mean(QR_UwRe_cov(:,:,end,:),4);

mean_QR_SwNr = mean(QR_SwNr,2);
cov_QR_SwNr = cov(QR_SwNr');
est_QR_SwNr_cov = mean(QR_SwNr_cov,3);

mean_QR_SwRe = mean(QR_SwRe(:,end,:),3);
cov_QR_SwRe = cov(squeeze(QR_SwRe(:,end,:))');
est_QR_SwRe_cov = mean(QR_SwRe_cov(:,:,end,:),4);

true_QR = [Q(tril(ones(nw))==1);R(tril(ones(nv))==1)];

[true_QR,mean_QR_UwNr,mean_QR_UwRe,mean_QR_SwNr,mean_QR_SwRe]

[diag(cov_QR_UwNr),diag(cov_QR_UwRe),diag(cov_QR_SwNr),diag(cov_QR_SwRe)]

[norm(cov_QR_UwNr-est_QR_UwNr_cov)/norm(est_QR_UwNr_cov),...
 norm(cov_QR_UwRe-est_QR_UwRe_cov)/norm(est_QR_UwRe_cov),...
 norm(cov_QR_SwRe-est_QR_SwRe_cov)/norm(est_QR_SwRe_cov),...
 norm(cov_QR_SwNr-est_QR_SwNr_cov)/norm(est_QR_SwNr_cov)]

[mean(time_UwNr), mean(time_UwRe), mean(time_SwNr), mean(time_SwRe);...
 mean(time_UwNr)/mean(time_UwNr), mean(time_UwRe)/mean(time_UwNr), mean(time_SwNr)/mean(time_UwNr), mean(time_SwRe)/mean(time_UwNr)]
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End: Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 1
figure(1)
steps=L:Number;

mean_All_QR_UwRe = mean(QR_UwRe,3);
var_All_QR_UwRe = var(QR_UwRe,0,3);
var_All_QR_UwRe(var_All_QR_UwRe<1e-10) = nan; 

mean_All_QR_SwRe = mean(QR_SwRe,3);
var_All_QR_SwRe = var(QR_SwRe,0,3);
var_All_QR_SwRe(var_All_QR_SwRe<1e-10) = nan; 

subplot(1,2,1)
hold all
%plot(steps,squeeze(QR_UwRe(1,:,1:100:end))','-.','Color',[.9 .8 .8])
%plot(steps,squeeze(QR_SwRe(1,:,1:100:end))','-.','Color',[1 .9 .9])

p1=plot(steps,mean_All_QR_SwRe(1,:),'r','LineWidth',1.5);
p2=plot(steps,mean_All_QR_SwRe(1,:)+sqrt(var_All_QR_SwRe(1,:)),'--r');
plot(steps,mean_All_QR_SwRe(1,:)-sqrt(var_All_QR_SwRe(1,:)),'--r')

p3=plot(steps,mean_All_QR_UwRe(1,:),'b--','LineWidth',1.5);
p4=plot(steps,mean_All_QR_UwRe(1,:)+sqrt(var_All_QR_UwRe(1,:)),'b-.');
plot(steps,mean_All_QR_UwRe(1,:)-sqrt(var_All_QR_UwRe(1,:)),'b-.')

%plot(steps,squeeze(QR_UwRe(4,:,1:100:end))','-.','Color',[.9 .8 .9])
%plot(steps,squeeze(QR_SwRe(4,:,1:100:end))','-.','Color',[1 .9 1])

p5=plot(steps,mean_All_QR_SwRe(4,:),'m','LineWidth',1.5);
p6=plot(steps,mean_All_QR_SwRe(4,:)+sqrt(var_All_QR_SwRe(4,:)),'--m');
plot(steps,mean_All_QR_SwRe(4,:)-sqrt(var_All_QR_SwRe(4,:)),'--m')

p7=plot(steps,mean_All_QR_UwRe(4,:),'--','LineWidth',1.5,'color',[0 0.7 0]);
p8=plot(steps,mean_All_QR_UwRe(4,:)+sqrt(var_All_QR_UwRe(4,:)),'-.','color',[0 0.7 0]);
plot(steps,mean_All_QR_UwRe(4,:)-sqrt(var_All_QR_UwRe(4,:)),'-.','color',[0 0.7 0])

grid on
xlim([L+N Number])
ylim([-2 6])
legend([p3,p4,p1,p2,p7,p8,p5,p6],{'UwRe - MC sample mean Q', 'UwRe - MC sample std Q',...
                                  'SwRe - MC sample mean Q', 'SwRe - MC sample std Q',...
                                  'UwRe - MC sample mean R(2,2)', 'UwRe - MC sample std R(2,2)',...
                                  'SwRe - MC sample mean R(2,2)', 'SwRe - MC sample std R(2,2)'},'NumColumns',2,'Interpreter','tex')
xlabel('Time steps','Interpreter','tex')   

subplot(1,2,2)
hold all
%plot(steps,squeeze(QR_UwRe(2,:,1:100:end))','-.','Color',[.8 .8 .9])
%plot(steps,squeeze(QR_SwRe(2,:,1:100:end))','-.','Color',[.9 .9 1])

p1=plot(steps,mean_All_QR_SwRe(2,:),'r','LineWidth',1.5);
p2=plot(steps,mean_All_QR_SwRe(2,:)+sqrt(var_All_QR_SwRe(2,:)),'--r');
plot(steps,mean_All_QR_SwRe(2,:)-sqrt(var_All_QR_SwRe(2,:)),'--r')

p3=plot(steps,mean_All_QR_UwRe(2,:),'b--','LineWidth',1.5);
p4=plot(steps,mean_All_QR_UwRe(2,:)+sqrt(var_All_QR_UwRe(2,:)),'b-.');
plot(steps,mean_All_QR_UwRe(2,:)-sqrt(var_All_QR_UwRe(2,:)),'b-.')

%plot(steps,squeeze(QR_UwRe(3,:,1:100:end))','-.','Color',[.8 .9 .8])
%plot(steps,squeeze(QR_SwRe(3,:,1:100:end))','-.','Color',[.9 1 .9])

p5=plot(steps,mean_All_QR_SwRe(3,:),'m','LineWidth',1.5);
p6=plot(steps,mean_All_QR_SwRe(3,:)+sqrt(var_All_QR_SwRe(3,:)),'--m');
plot(steps,mean_All_QR_SwRe(3,:)-sqrt(var_All_QR_SwRe(3,:)),'--m')

p7=plot(steps,mean_All_QR_UwRe(3,:),'--','LineWidth',1.5,'color',[0 0.7 0]);
p8=plot(steps,mean_All_QR_UwRe(3,:)+sqrt(var_All_QR_UwRe(3,:)),'-.','color',[0 0.7 0]);
plot(steps,mean_All_QR_UwRe(3,:)-sqrt(var_All_QR_UwRe(3,:)),'-.','color',[0 0.7 0])

grid on
xlim([L+N Number])
ylim([-4 5])
legend([p3,p4,p1,p2,p7,p8,p5,p6],{'UwRe - MC sample mean R(1,1)', 'UwRe - MC sample std R(1,1)',...
                                  'SwRe - MC sample mean R(1,1)', 'SwRe - MC sample std R(1,1)',...
                                  'UwRe - MC sample mean R(1,2)', 'UwRe - MC sample std R(1,2)',...
                                  'SwRe - MC sample mean R(1,2)', 'SwRe - MC sample std R(1,2)'},'NumColumns',2,'Interpreter','tex')
xlabel('Time steps','Interpreter','tex')              
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End: Plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plots II %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 1
circle = [sin(linspace(0,2*pi,80));cos(linspace(0,2*pi,80))];
figure(2)
for i=1:floor(nQRu/2)
subplot(1,floor(nQRu/2),i)
hold all
grid on
if i==1
    xlim([2.6 3.4])
    ylim([1.65 2.35])
    xlabel('Q','Interpreter','tex')
    ylabel('R(1,1)','Interpreter','tex')
else
    xlim([-1.3 -0.7])
    ylim([0.7 1.3])
    xlabel('R(1,2)','Interpreter','tex')
    ylabel('R(2,2)','Interpreter','tex')
end

in=(1:2)+(i-1)*2;

%{
%%% MC Estimates
plot(QR_UwNr(in(1),1:1e1:end),QR_UwNr(in(2),1:1e1:end),'x','MarkerSize',2,'color',[.8 .8 1]);
plot(squeeze(QR_UwRe(in(1),end,1:1e1:end)),squeeze(QR_UwRe(in(2),end,1:1e1:end)),'square','MarkerSize',2,'color',[1 .8 1]);
plot(QR_SwNr(in(1),1:1e1:end),QR_SwNr(in(2),1:1e1:end),'+','MarkerSize',2,'color',[.6 1 .8]);
plot(squeeze(QR_SwRe(in(1),end,1:1e1:end)),squeeze(QR_SwRe(in(2),end,1:1e1:end)),'o','MarkerSize',2,'color',[1 .8 .8]);
%%%
%}

p0 = plot(true_QR(in(1)),true_QR(in(2)),'ko','MarkerSize',20,'LineWidth',0.5);

p1 = plot(mean_QR_UwNr(in(1)),mean_QR_UwNr(in(2)),'b+','MarkerSize',15);
elips_QR_UwNr = mean_QR_UwNr(in) + chol(cov_QR_UwNr(in,in))'*circle;
p2 = plot(elips_QR_UwNr(1,:),elips_QR_UwNr(2,:),'b-','LineWidth',1.2);
est_elips_QR_UwNr = mean_QR_UwNr(in) + chol(est_QR_UwNr_cov(in,in))'*circle;
p3 = plot(est_elips_QR_UwNr(1,:),est_elips_QR_UwNr(2,:),'b--','LineWidth',1.2);

p4 = plot(mean_QR_UwRe(in(1)),mean_QR_UwRe(in(2)),'msquare','MarkerSize',15);
elips_QR_UwRe = mean_QR_UwRe(in) + chol(cov_QR_UwRe(in,in))'*circle;
p5 = plot(elips_QR_UwRe(1,:),elips_QR_UwRe(2,:),'m-.','LineWidth',1.2);
est_elips_QR_UwRe = mean_QR_UwRe(in) + chol(est_QR_UwRe_cov(in,in))'*circle;
p6 = plot(est_elips_QR_UwRe(1,:),est_elips_QR_UwRe(2,:),'m:','LineWidth',1.2);

p7 = plot(mean_QR_SwNr(in(1)),mean_QR_SwNr(in(2)),'x','color',[0 .5 .2],'MarkerSize',13);
elips_QR_SwNr = mean_QR_SwNr(in) + chol(cov_QR_SwNr(in,in))'*circle;
p8 = plot(elips_QR_SwNr(1,:),elips_QR_SwNr(2,:),'-','color',[0 .5 .2],'LineWidth',1.2);
est_elips_QR_SwNr = mean_QR_SwNr(in) + chol(est_QR_SwNr_cov(in,in))'*circle;
p9 = plot(est_elips_QR_SwNr(1,:),est_elips_QR_SwNr(2,:),'--','color',[0 .5 .2],'LineWidth',1.2);

p10 = plot(mean_QR_SwRe(in(1)),mean_QR_SwRe(in(2)),'rd','MarkerSize',10);
elips_QR_SwRe = mean_QR_SwRe(in) + chol(cov_QR_SwRe(in,in))'*circle;
p11 = plot(elips_QR_SwRe(1,:),elips_QR_SwRe(2,:),'r-.','LineWidth',1.2);
est_elips_QR_SwRe = mean_QR_SwRe(in) + chol(est_QR_SwRe_cov(in,in))'*circle;
p12 = plot(est_elips_QR_SwRe(1,:),est_elips_QR_SwRe(2,:),'r:','LineWidth',1.2);
end

pn = plot(0,nan,'w.');

legend([p0,p1,p4,p7,p10,pn,p2,p5,p8,p11,pn,p3,p6,p9,p12],{'True', 'UwNr - MC sample mean', 'UwRe - MC sample mean', 'SwNr - MC sample mean', 'SwRe - MC sample mean',...
                                                          '    ', 'UwNr - MC sample std', 'UwRe - MC sample std', 'SwNr - MC sample std', 'SwRe - MC sample std',...
                                                          '    ', 'UwNr - average est std', 'UwRe - average est std', 'SwNr - average est std', 'SwRe - average est std'},'NumColumns',3)                                                
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End: Plots II %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%save('Example_recursion.mat')
