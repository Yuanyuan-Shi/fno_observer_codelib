N=50; % spatial discretization
nsteps=100; % temporal discretization

L=1; % length of spatial domain
timeperiod=2; % length of time domain
t0=0; % initial time
T=2; % prescribed convergence time

d=L/(N-1); % spatial grid step size
tau=(timeperiod-t0)/nsteps; % temporal grid step size

mu0=5; % controller initial gain
mu0_tilde=10; % observer initial gain

eps=1; % real diffusivity for heat equation
lambda = 12;

x=linspace(0,L,N); %generate x grid points
y=linspace(0,L,N); %generate y grid points
t=linspace(t0,timeperiod,nsteps); %generate t grid points

init_cond=5*sin(x(1:N-2).*(pi)); %set initial condition

restart=round(0.7*T/tau); %Set time at which the time-varying controller restarts
terms=12; %number of terms to use in kernel approximation, m=3

%Plotting inputs
interp_stepst=60;
interp_stepsx=60;
t_plotting=linspace(0,timeperiod,interp_stepst);
x_plotting=linspace(0,L,interp_stepsx);

%% Generate controller backstepping kernels (time-varying & time-invariant)
k1=zeros(N,nsteps);
% for k=1:restart
%     nu=(1-(t(k)-t0)/T);
%     factor=-1/eps*mu0/(nu^2);
%     for j=1:N-1
%         theta=1/eps*(1-y(j)^2)/(4*T*nu);
%         arg=sqrt(1/eps*mu0*(1-y(j)^2)/(nu^2));
%         k1(j,k)=factor*exp(theta)*besseli(1,arg)/arg;
%     end
% end

k2=zeros(N,1);
% for j=1:N-1 % Andrey's kernel gains for exp. stabilization
%     k2(j)=-mu0/eps*besseli(1,sqrt(mu0/eps*(1-y(j)^2)))/sqrt(mu0/eps*(1-y(j)^2));
% end

%% Generate observer backstepping kernels
p1=zeros(N,nsteps);
for k=1:restart % m=3 for prescribed-time estimation
    nu=(1-(t(k)-t0)/T);
    for j=1:N-1
        factor=-mu0_tilde*x(j)/(2*eps*nu^3);
        p1(j,k)=factor*sum_3(-mu0_tilde,T,eps,t(k),t0,terms,x(j),1);
    end
end

p2=zeros(N,1);
for j=1:N-1 % Andrey's kernel gains for exponential estimation
    p2(j)=-x(j)*lambda*besseli(1,sqrt(lambda*(1-x(j)^2)))/sqrt(lambda*(1-x(j)^2));
end

%% Simulate observer/output feedback dynamics via implicit Euler method
alpha=-eps*tau/d^2;
beta=eps*tau/d;
gamma=lambda*tau;
A1=zeros(2*N-4); %This is the matrix A within linear equation AT(:,i+1)=T(:,i)
for i=2:2*N-5 %load A & B at interior points
    A1(i,i-1)=alpha;
    A1(i,i)=1-2*alpha-gamma;
    A1(i,i+1)=alpha;
end
A1(1,1)=1-2*alpha-gamma; A1(1,2)=alpha;
A1(N-2,N-1)=0;
A1(N-1,N-2)=0;
A1(end,end-1)=alpha; A1(end,end)=1-2*alpha-gamma;
A1_reset1=A1(N-2,:);
A1_reset2=A1(2*N-4,:);
A1_reset3=A1(:,N-2);
A1_reset4=A1(:,2*N-4);

A2=A1;
A2_reset1=A2(N-2,:);
A2_reset2=A2(2*N-4,:);
A2_reset3=A2(:,N-2);
A2_reset4=A2(:,2*N-4);

v1_total=zeros(2*N-4,nsteps); % simulate plant and observer systems (state dimension doubles)
v1_total(1:N-2,1)=init_cond;
v1=zeros(N-2,nsteps);
v1_obs=zeros(N-2,nsteps);
error1=zeros(N-2,nsteps);
error1(:,1)=v1_total(1:N-2,1)-v1_total(N-1:2*N-4,1);

v2_total=zeros(2*N-4,nsteps);
v2_total(1:N-2,1)=init_cond;
v2_obs=zeros(N-2,nsteps);
error2=zeros(N-2,nsteps);
error2(:,1)=v2_total(1:N-2,1)-v2_total(N-1:2*N-4,1);

A2(N-2,:)=A2_reset1+[zeros(1,N-2) transpose(k2(2:N-1)).*(alpha*d)];
A2(2*N-4,:)=A2_reset2+[zeros(1,N-2) transpose(k2(2:N-1)).*(alpha*d)];
A2(:,N-2)=A2_reset3+[zeros(N-2,1);p2(2:N-1).*beta];
A2(:,2*N-4)=A2_reset4+[zeros(N-2,1);-p2(2:N-1).*beta];
runtime=2;
for mtime=2:nsteps
    if runtime<restart
        runtime=runtime+1;
    end
    A1(N-2,:)=A1_reset1+[zeros(1,N-2) transpose(k1(2:N-1,runtime)).*(alpha*d)];
    A1(2*N-4,:)=A1_reset2+[zeros(1,N-2) transpose(k1(2:N-1,runtime)).*(alpha*d)];
    A1(:,N-2)=A1_reset3+[zeros(N-2,1);p1(2:N-1,runtime).*beta];
    A1(:,2*N-4)=A1_reset4+[zeros(N-2,1);-p1(2:N-1,runtime).*beta];

    v1_total(:,mtime)=A1\v1_total(:,mtime-1);
    v1(:,mtime)=v1_total(1:N-2,mtime);
    v1_obs(:,mtime)=v1_total(N-1:2*N-4,mtime);
    error1(:,mtime)=v1(:,mtime)-v1_obs(:,mtime);
  
    v2_total(:,mtime)=A2\v2_total(:,mtime-1);
    v2(:,mtime)=v2_total(1:N-2,mtime);
    v2_obs(:,mtime)=v2_total(N-1:2*N-4,mtime);
    error2(:,mtime)=v2(:,mtime)-v2_obs(:,mtime);    
end

v1norm=zeros(1,nsteps);
for k=1:nsteps
    v1norm(k)=sqrt(v1(:,k)'*v1(:,k).*d);
end

v2norm=zeros(1,nsteps);
for k=1:nsteps
    v2norm(k)=sqrt(v2(:,k)'*v2(:,k).*d);
end

error1norm=zeros(1,nsteps);
for k=1:nsteps
    error1norm(k)=sqrt(error1(:,k)'*error1(:,k).*d);
end

error2norm=zeros(1,nsteps);
for k=1:nsteps
    error2norm(k)=sqrt(error2(:,k)'*error2(:,k).*d);
end

a=round(T/tau);
figure
semilogy(t(1:a),error1norm(1:a),'linewidth',2.5,'color','r');
hold on
semilogy(t(1:a),error2norm(1:a),'linewidth',2.5,'color','b');
xlabel('$t$','interpreter','latex','fontsize',30)
ylabel('$||\tilde{u}||_{L^2}$','interpreter','latex','fontsize',30);
lgd2=legend('Prescribed-time estimation','Exponential estimation');
lgd2.FontSize=20;
grid on;