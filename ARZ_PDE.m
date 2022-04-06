clear
clc

%Parameter values
vm = 40; % m/s 100 miles per hour
rm = 0.16; %  240 veh/mile /m
tau = 60; % 
L = 500; % m 1km
T = 240; % second /hour
vs = 10; % m/s 20 miles/hour
rs =0.12 ; %150 veh/mile

qs = rs * vs ; % 1.5 per second
gam= 1;
ps=vm/rm * qs/vs;

%Differentiation in space and time
dx= 5 ;
dt=0.025;
t=0:dt:T;
x=0:dx:L;
M=length(x);
N=length(t);

%Define variables
bv = zeros(M,N);
w = zeros(M,N);


K = zeros(M,M);
YK = zeros(M,1);
YL = zeros(M,1);



% charateristics
lambda_1 = vs ;
lambda_2 = vs - rs * vm/rm ;



% initial conditions
w(:,1) = qs * 0.1 * sin(3*x/L* pi)'  - qs /(lambda_1 - lambda_2) * vs * 0.1 * sin(3*x/L* pi)' ;
bv(:,1) =  - qs/(lambda_1 - lambda_2) * vs * 0.1 * sin(3*x/L* pi)';


%time evolution
for n= 1 : N-1
    
   % q(:,n)=r(:,n).*v(:,n);  
    for j= 2 : M-1
          
         %U_mid
         w_pmid = 1/2 * (w(j+1,n) + w(j,n))...
             - dt/dx/2 * vs * (w(j+1,n) - w(j,n)) - 1/4 * dt /tau * (w(j+1,n) + w(j,n)) ;
         v_pmid = 1/2 * (bv(j+1,n) + bv(j,n))...
              - dt/dx/2 * lambda_2 * (bv(j+1,n) - bv(j,n)) - 1/4 * dt /tau * (w(j+1,n) + w(j,n));
          
         w_mmid = 1/2 * (w(j-1,n) + w(j,n))...
             - dt/dx/2 * vs * (w(j,n) - w(j-1,n)) - 1/4 * dt /tau * (w(j-1,n) + w(j,n)) ;
         v_mmid = 1/2 * (bv(j-1,n) + bv(j,n))...
              - dt/dx/2 * lambda_2 * (bv(j,n) - bv(j-1,n)) - 1/4 * dt /tau * (w(j-1,n) + w(j,n));
         
         %final
         w(j,n+1)= w(j,n) - dt/dx * vs * ( w_pmid - w_mmid ) - 1/2 * dt /tau * (w_pmid + w_mmid);
         bv(j,n+1)= bv(j,n) - dt/dx * lambda_2 * ( v_pmid - v_mmid ) - 1/2 * dt /tau * (w_pmid + w_mmid);
      
                  
         %intergral inside UORM controller
        YK(j) = K(M,j)* exp((j-1)*dx/(tau*vs))* w(j,n+1);
    end
    
    
   w(M,n+1) = w(M,n) - dt/dx * vs * (w(M,n) - w(M-1,n)) - dt/tau * w(M,n);
   bv(1,n+1) = bv(1,n) -  dt/dx * lambda_2 * (bv(2,n) - bv(1,n)) - dt /tau * w(1,n);
   
  % Boundary conditions
  w(1,n+1) = lambda_2 / lambda_1 * bv(1,n+1);  
  bv(M,n+1) = w(M,n+1);

  


end

tv = (lambda_1 - lambda_2)/qs * bv;
v = (tv + vs * ones(M,N));

tq = w - rs * lambda_2 / qs * bv;
r = (tq + qs * ones(M,N))./v;


figure
colormap([0  0  0]);
subplot(1,2,1);
surf(r,'FaceColor','white','EdgeColor','interp','MeshStyle','row');
axis([0, N, 0, M, 0.08, 0.16]);
ylabel('Position x (m)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('Density (vehicle/km)','fontsize', 26)
 ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:2400:9600;
 set(ax,'XTickLabel',[0,1,2,3,4])
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:20:100;
 set(ax,'YTickLabel',[0,100,200,300,400,500])
 ax.ZTick = 0.08:0.01:0.16;
 set(ax,'ZTickLabel',[80,90,100,110,120,130,140,150,160],'fontsize', 22)
 hold on
 axis([0, N, 0, M, 0.08, 0.16]);
 x1=0:1:100;
 plot3(0*ones(1,numel(x1)),x1,r(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1:9600;
 plot3(x2,M*ones(1,numel(x2)),r(M,:),'-r','LineWidth',4);
 
 
colormap([0  0  0]); 
subplot(1,2,2);
surf(v,'FaceColor','white','edgecolor', 'interp','MeshStyle','row');
axis([0, N, 0, M, 7, 12.5]);
ylabel('Position x (m)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('velocity (m/s)','fontsize', 26)
ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:2400:9600;
 set(ax,'XTickLabel',[0,1,2,3,4])
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:20:100;
 set(ax,'YTickLabel',[0,100,200,300,400,500])
 ax.ZTick =7:1:12;
 set(ax,'ZTickLabel',[7,8,9,10,11,12],'fontsize', 22)
 hold on
 x1=0:1:100;
 axis([0, N, 0, M, 7, 12.5]);
 plot3(0*ones(1,length(x1)),x1,v(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1:9600;
 plot3(x2,M*ones(1,numel(x2)),v(M,:),'-r','LineWidth',4);
