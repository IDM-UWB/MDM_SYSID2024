function [A2u,covRes,Mat_covRes,Xi_A2] = MDM(L,N,F,G,E,nz,H,D,z,u)
% Author: Oliver Kost, kost@ntis.zcu.cz
%
% Requires files: O_Gamma.m
%
% L: number of measuremnts in Z; User parameter
% N: Nth-prediction of Z; User parameter

[nx,nw] = size(E{1});
nv = size(D{1},2);
Number = size(H,1);

%%% Replikation matrix for noise covarainces
Psi=zeros(((L+N-1)*nw+(L+N)*nv)^2,nw*(nw+1)/2+nv*(nv+1)/2);
Fq = nan(nw); Fq(tril(ones(nw))==1) = 1:nw*(nw+1)/2; Fq(triu(ones(nw),1)==1) = Fq(tril(ones(nw),-1)==1);
for j=1:nw*(nw+1)/2
    Psi(:,j)=reshape(blkdiag(kron(eye(L+N-1),Fq==j),zeros((L+N)*nv)),((L+N-1)*nw+(L+N)*nv)^2,1);
end
Fr = nan(nv); Fr(tril(ones(nv))==1) = 1:nv*(nv+1)/2; Fr(triu(ones(nv),1)==1) = Fr(tril(ones(nv),-1)==1);
for j=1:nv*(nv+1)/2
    Psi(:,nw*(nw+1)/2+j)=reshape(blkdiag(zeros((L+N-1)*nw),kron(eye(L+N),Fr==j)),((L+N-1)*nw+(L+N)*nv)^2,1);
end
%%% End: Replikation matrix for noise covarainces

O = cell(Number-L+1,1); % Obsevable matrix
Gamma = cell(Number-L+1,1);% Gamma matrix

Awu = cell(Number-L+1,1);
Avz = cell(Number-L+1,1);
A2 = cell(Number-L+1,1);

Xi = cell(Number-L+1,1);
Xi_A2 = cell(Number-L+1,1);
A2u=cell(Number-L+1,1);
for t=1+N:Number-L+1
    if isempty(O{t-N})
        [O{t-N},Gamma{t-N}] = O_Gamma(F,H,nz,L,t-N);
    end
    if N~=0
        [O{t},Gamma{t}] = O_Gamma(F,H,nz,L,t);
    end
         
    if rank(O{t-N})<nx % Observability check
        disp('Set larger parameter L or N')
        error('MDM: The state is UNobservable')
    end
    
    F_N = 1;
    Phi = zeros(nx,0);
    for i=0:N-1
        F_N = F{t-N+i} * F_N;
        
        Phi_part=eye(nx);
        for j=0:i-1
            Phi_part = F{t-i+j} * Phi_part; 
        end
        Phi = [Phi_part,Phi];
    end
    
    C = O{t}*F_N*((O{t-N}'*O{t-N})\O{t-N}');
    
    nOt=sum(nz(t:(t+L-1)));
    ntLN=sum(nz(t+L-1-N+1:(t+L-1)));
    ntNL=sum(nz((t-N):(t-1)));
    
    Awu{t} = [eye(nOt),eye(nOt)]*...
             [O{t}*Phi     , Gamma{t};...
             -C*Gamma{t-N} , zeros(nOt,N*nx)];
    Avz{t} = [eye(nOt),eye(nOt)]*...
             [zeros(nOt,ntNL) , eye(nOt);...
             -C               , zeros(nOt,ntLN)];
    
    A = [Awu{t}*blkdiag(E{t-N:t+L-2}), Avz{t}*blkdiag(D{t-N:t+L-1})]; % [Aw, Av]

    A2{t} = kron(A,A); 
    
    nA2 = sum(nz(t:(t+L-1)))^2; %*0+size(A2{t},1);
    Xi_part = eye(nA2);
    Xi{t} = Xi_part(1==triu(ones(sqrt(nA2))),:);

    Xi_A2{t} = Xi{t}*A2{t};
    
    A2u{t} = Xi_A2{t}*Psi;
end

covRes=cell(Number-L+1,1); % Covariance of residue vector
for t=1+N:Number-L+1
    Z = vertcat(z{(t-N):(t+L-1)}); % Augmented measurement vector
    U = vertcat(u{(t-N):(t+L-2)}); % Augmented control vector
    Res = Avz{t} * Z - Awu{t} * blkdiag(G{t-N:t+L-2}) * U; 
    covRes{t} = Xi{t}*reshape(Res*Res',size(Res,1)^2,1); 
end

Mat_covRes{1} = Awu;
Mat_covRes{2} = Avz;
Mat_covRes{3} = Xi;
end

