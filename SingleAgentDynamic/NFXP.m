clear

beta = 0.975;

% True parameter
RC = 11.7257; theta11 = 2.4569; theta30 = 0.0937;
theta31 = 0.4475; theta32 = 0.4459; theta33 = 0.0127;
theta34 = 0.0002;

% Packaging

THETA_TRUE = [RC,theta11,theta30,theta31,theta32,theta33,theta34]';

M = 175 ;% Dimension of X
XMAX = 5000;
N = 5;
T = 120;

% Data Generating
[x,a] = func_data(M,XMAX,N,T,THETA_TRUE,beta);

%% Initial values for parameters
RC = 12; theta11 = 3;
THETA3 = func_theta3(M,XMAX,T,x,a);
% Estimate THETA3
THETA(:,1) = [RC,theta11,THETA3']';

%% Outer-loop (updating parameters)

tic
kMAX = 200;
logL = ones(1,kMAX)*(-1e+12); % initial value for log-likelihood

for k =1:kMAX
    fprintf('NFXP:   k = %d, logL = %f\n',k,logL(k));

    %% Inner-loop (find the fixed point)
    PO(:,1) = ones(M,1)*0.6;
    iterMAX = 100; % Inner-loop Maximum Iteration
    ERROR = 1e-12;
    X = linspace(0,XMAX,M)':
    for iter = 1:iterMAX
        V(:,iter) = func_Phi(X,M,PO(:,iter),THETA(:,k),beta);
        PO(:,iter+1) = func_Lambda(X,M,V(:,iter),THETA(:,k),beta);
        if(pdist([PO(:,iter),PO(:,iter+1)]')<ERROR)
            break;
        end
    end
    %% End of Inner-loop
    