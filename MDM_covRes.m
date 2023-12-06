function covRes = MDM_covRes(L,N,G,z,u,Mat_covRes)
% Author: Oliver Kost, kost@ntis.zcu.cz
%
% L: number of measuremnts in Z; User parameter
% N: Nth-prediction of Z; User parameter

Number = size(z,1);

Awu = Mat_covRes{1};
Avz = Mat_covRes{2};
Xi = Mat_covRes{3};

covRes=cell(Number-L+1,1);
for t=1+N:Number-L+1
    Z = vertcat(z{(t-N):(t+L-1)}); % Augmented measurement vector
    U = vertcat(u{(t-N):(t+L-2)}); % Augmented control vector
    Res = Avz{t} * Z - Awu{t} * blkdiag(G{t-N:t+L-2}) * U;
    covRes{t} = Xi{t}*reshape(Res*Res',size(Res,1)^2,1);
end

end





