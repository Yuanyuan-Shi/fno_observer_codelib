clear
clc

%Parameter values
vm = 40; % m/s 100 miles per hour
rm = 0.16; %  240 veh/mile /m
tau = 60; % 3s
L = 500; % m 1km
T = 240; % second /hour


%Differentiation in space and time
dx= 5;
dt= 0.2;
t=0:dt:T;
x=0:dx:L;
M=length(x);
N=length(t);

num = 1200;
obs_data = zeros(num, M, N, 2);
sys_data = zeros(num, M, N, 2);

for n_sample = 1:num
    %% Set reference velocity and density
%     vs = 10; % m/s 20 miles/hour
%     rs = 0.12 ; %150 veh/mile
%     vs = 10; % m/s 20 miles/hour
%     rs = 0.1114; %150 veh/mile
    
    vs = 10 + 2*rand(1); %5 + 5*rand(1); %refernce velocity
    rs = 0.11 + 0.01*rand(1); %0.08 + (0.12-0.08)*rand(1); %refefrence density
    
    qs = rs * vs ; % 1.5 per second
    gam= 1;
    ps=vm/rm * qs/vs;
    ys = 0;

    %Define variables
    r = zeros(M,N);
    y = zeros(M,N);
    r_mid = zeros(M,1);
    y_mid = zeros(M,1);

    hr = zeros(M,N);
    hy = zeros(M,N);
    hr_mid = zeros(M,1);
    hy_mid = zeros(M,1);
    Y_c = zeros(1,N);


    % charateristics
    lambda_1 = vs ;
    lambda_2 = vs - rs * vm/rm ;

    % offset
    scale = rand(1);

    % initial conditions IC/BC consistent
    r(:,1) = rs * (sin(3 * (x/L-scale) * pi) * 0.1 + ones(1,M))';
    y(:,1) = vs * r(:,1) .* (-sin(3 * (x/L-scale) * pi) * 0.1 + ones(1,M))' - vm * r(:,1) + vm / rm * r(:,1).^2 ;

    disp([n_sample, rs, vs, scale])
    
    % intial condition for estimator 
    hr(:,1) = rs * ones(1,M)';
    hy(:,1) = vs * hr(:,1) .* ( ones(1,M))' - vm * hr(:,1) + vm / rm * hr(:,1).^2 ;


    % Fundamental diagram (Greenshields)
    Veq = @(rho) vm * (1 - rho/rm);

    % Fundamental diagram (Greenshields)
    QpR = @(rho) rho* vm * (1 - rho/rm);

    % flux
    F_r = @(rho, y)  y + rho * Veq(rho);
    F_y = @(rho, y)  y * (y/rho + Veq(rho));


    % spatial function
    c_x = @ (x) - 1 /tau * exp(-x/tau/vs);
    r_x = @ (x) - lambda_1 /(lambda_1 - lambda_2) * c_x(-lambda_2 /(lambda_1 - lambda_2)*(L-x));
    s_x = @ (x) lambda_1 /(lambda_1 - lambda_2) *  c_x(lambda_1 /(lambda_1 - lambda_2)* x - lambda_2 /(lambda_1 - lambda_2)* L);


    % initial conditions
    w(:,1) = zeros(M,1);
    bv(:,1) = zeros(M,1);
    Y_c(1) = y(M,1) - qs + r(M,1) * Veq(r(M,1)) + rs * lambda_2/(lambda_1 - lambda_2) * (y(M,1) /r(M,1)  + Veq(r(M,1)) - vs);
    bv(M,1) = Y_c(1);


    % related parameters 
    yr = 2 * vm /rm * rs - vm;

    %%%%%%%%%%%%%%%%%%%%%Nonlinear ARZ /  linearized ARZ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Lax-Wendroff's method
    for n = 1 : N-1

      v_temp=  y(M,n) /r(M,n)  + Veq(r(M,n)); 
      Y_c(n) = y(M,n) + r(M,n) * Veq(r(M,n)) - qs + rs * lambda_2/(lambda_1 - lambda_2) * (v_temp - vs);
      Y_c(n) = exp(-L/tau/lambda_1) *  Y_c(n+1);

      hv_temp = hy(M,n) /hr(M,n)  + Veq(hr(M,n));
      w(M-1,n) =  exp(-L/tau/lambda_1) * ( hy(M,n) + hr(M,n) * Veq(hr(M,n)) - qs + rs * lambda_2/(lambda_1 - lambda_2) * (hv_temp - vs));

      w_L = Y_c(n)- w(M-1,n);

       for j = 2 : M-1

    % linear gain
       G_r = r_x((j-1) * dx) * w_L;       
       G_s = s_x((j-1) * dx) * w_L;

    %Nonlinear observer
           xi1p = G_r * exp(-(j)* dx / tau / vs);
           xi2  = G_s ; 

           S_hrp = (xi1p - xi2)/vs;
           S_hyp = xi1p - lambda_2 / lambda_1 * xi2  + QpR(rs) *  S_hrp;


           hr_pmid = 1/2 * (hr(j+1,n) + hr(j,n)) ...
               - dt/(2 * dx) * ( F_r(hr(j+1,n), hy(j+1,n)) - F_r(hr(j,n), hy(j,n)))...
               + 1/2 * dt *  S_hrp;

           hy_pmid = 1/2 * (hy(j+1,n) + hy(j,n)) ...
               - dt/(2 * dx) * ( F_y(hr(j+1,n), hy(j+1,n)) - F_y(hr(j,n), hy(j,n))) ...
               - 1/4 * dt /tau * (hy(j+1,n) + hy(j,n))...
               +  1/2 * dt * S_hyp;

           xi1m = G_r * exp(-(j-2) * dx/ tau / vs);

           S_hrm =  (xi1m-xi2)/vs;
           S_hym = xi1m - lambda_2 / lambda_1 * xi2  + QpR(rs) * (xi1m-xi2)/vs;

           hr_mmid = 1/2 * (hr(j-1,n) + hr(j,n)) ...
               - dt/(2 * dx) * ( F_r(hr(j,n), hy(j,n)) - F_r(hr(j-1,n), hy(j-1,n)))...
              + 1/2 * dt * S_hrm;

           hy_mmid = 1/2 * (hy(j-1,n) + hy(j,n)) ...
               - dt/(2 * dx) * ( F_y(hr(j,n), hy(j,n)) - F_y(hr(j-1,n), hy(j-1,n))) ...
               - 1/4 * dt /tau * (hy(j-1,n) + hy(j,n))...
                +  1/2 * dt * S_hym;


            %final
           hr(j,n+1) = hr(j,n) - dt/dx * (F_r(hr_pmid,hy_pmid) - F_r(hr_mmid,hy_mmid))...
               + 1/2 * dt * (S_hrp + S_hrm) ;
           hy(j,n+1) = hy(j,n) - dt/dx * (F_y(hr_pmid,hy_pmid) - F_y(hr_mmid,hy_mmid))...
               - 1/2 * dt/tau * (hy_pmid + hy_mmid) ...
               + 1/2 * dt * (S_hyp + S_hym) ;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

           % U_mid

           r_pmid = 1/2 * (r(j+1,n) + r(j,n)) ...
               - dt/(2 * dx) * ( F_r(r(j+1,n), y(j+1,n)) - F_r(r(j,n), y(j,n)));

           y_pmid = 1/2 * (y(j+1,n) + y(j,n)) ...
               - dt/(2 * dx) * ( F_y(r(j+1,n), y(j+1,n)) - F_y(r(j,n), y(j,n))) ...
               - 1/4 * dt /tau * (y(j+1,n) + y(j,n));

           r_mmid = 1/2 * (r(j-1,n) + r(j,n)) ...
               - dt/(2 * dx) * ( F_r(r(j,n), y(j,n)) - F_r(r(j-1,n), y(j-1,n)));

           y_mmid = 1/2 * (y(j-1,n) + y(j,n)) ...
               - dt/(2 * dx) * ( F_y(r(j,n), y(j,n)) - F_y(r(j-1,n), y(j-1,n))) ...
               - 1/4 * dt /tau * (y(j-1,n) + y(j,n));

            %final
           r(j,n+1) = r(j,n) - dt/dx * (F_r(r_pmid,y_pmid) - F_r(r_mmid,y_mmid));
           y(j,n+1) = y(j,n) - dt/dx * (F_y(r_pmid,y_pmid) - F_y(r_mmid,y_mmid))...
               - 1/2 * dt/tau * (y_pmid + y_mmid);

       end


    % Boundary conditions (characteristic variables)
    r(1,n+1) = r(2,n+1) ;
    y(1,n+1) = qs - r(1,n+1) * Veq(r(1,n+1));
    r(M,n+1) = rs;
    y(M,n+1) = y(M-1,n+1);

    %observer BC
      hr(1,n+1) = hr(2,n+1); 
      hy(1,n+1) = qs - hr(1,n+1) * Veq(hr(1,n+1));

      hr(M,n+1) = rs;
      hy(M,n+1) = y(M,n+1); 
    end

    v = y./r + Veq(r); 
    v(M,N) = v(M,N-1);


    hv = hy./hr + Veq(hr); 
    hv(M,N) = hv(M,N-1);
    
    FLAG = isnan(hy_mmid);
    if(FLAG)
        break;
    end
    
    sys_data(n_sample, :, :, 1) = r;
    sys_data(n_sample, :, :, 2) = v;
    obs_data(n_sample, :, :, 1) = hr;
    obs_data(n_sample, :, :, 2) = hv;
    
    
end

%plot of rho and hr
% index = 1;
% hold on
% subplot(1,2,1);
% a1 = plot(sys_data(index, :, N, 1));
% M1 = "Curve 1";
% a2 = plot(obs_data(index, :, N, 1)-sys_data(index, :, N, 1));
% M2 = "Curve 2";
% legend([a1,a2], [M1, M2]);

%% plot of rho and hr
index = 1;
figure
colormap([0  0  0]);
subplot(2,2,1);
surf(squeeze(sys_data(index, :, :, 1)),'FaceColor','white','EdgeColor','interp','MeshStyle','row');
axis([0, N, 0, M, 0.10, 0.14]);
ylabel('position x (m)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('density (vehicle/km)','fontsize', 26)
ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:T/dt/4:T/dt;
 set(ax,'XTickLabel',[0,1,2,3,4])
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:20:100;
 set(ax,'YTickLabel',[0,100,200,300,400,500])
%  ax.ZTick = 0.10:0.01:0.14;
%  set(ax,'ZTickLabel',[100,110,120,130,140],'fontsize', 22)
 hold on
 axis([0, N, 0, M, 0.0, 0.14]);
 x1=0:1:M-1;
 plot3(0*ones(1,numel(x1)),x1,r(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1:N-1;
 plot3(x2,M*ones(1,numel(x2)),r(M,:),'-r','LineWidth',4);
 
subplot(2,2,2);
surf(squeeze(obs_data(index, :, :, 1)),'FaceColor','white','EdgeColor','interp','MeshStyle','row');
axis([0, N, 0, M, 0.10, 0.14]);
ylabel('Position x (m)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('Density (vehicle/km)','fontsize', 26)
 ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:T/dt/4:T/dt;
 set(ax,'XTickLabel',[0,1,2,3,4])
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:20:100;
 set(ax,'YTickLabel',[0,100,200,300,400,500])
%  ax.ZTick = 0.10:0.01:0.14;
%  set(ax,'ZTickLabel',[100,110,120,130,140],'fontsize', 22)
 hold on
 axis([0, N, 0, M, 0.0, 0.14]);
 x1=0:1:M-1;
 plot3(0*ones(1,numel(x1)),x1,hr(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1:N-1;
 plot3(x2,M*ones(1,numel(x2)),hr(M,:),'-r','LineWidth',4);
 
 
% figure
colormap([0  0  0]); 
subplot(2,2,3);
surf(squeeze(sys_data(index, :, :, 2)),'FaceColor','white','edgecolor', 'interp','MeshStyle','row');
axis([0, N, 0, M, 8, 12]);
ylabel('position x (m)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('velocity (m/s)','fontsize', 26)
%title('ARZ v','fontsize', 22)
 ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:T/dt/4:T/dt;
 set(ax,'XTickLabel',[0,1,2,3,4])
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:20:100;
 set(ax,'YTickLabel',[0,100,200,300,400,500])
%  ax.ZTick = 8:1:12;
%  set(ax,'ZTickLabel',[8,9,10,11,12],'fontsize', 22)
 hold on
 x1=0:1:M-1;
axis([0, N, 0, M, 8, 16]);
 plot3(0*ones(1,length(x1)),x1,v(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1:N-1;
 plot3(x2,M*ones(1,numel(x2)),v(M,:),'-r','LineWidth',4);

colormap([0  0  0]); 
subplot(2,2,4);
surf(squeeze(obs_data(index, :, :, 2)),'FaceColor','white','edgecolor', 'interp','MeshStyle','row');
axis([0, N, 0, M, 8, 12]);
ylabel('Position x (m)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('velocity (m/s)','fontsize', 26)
ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:T/dt/4:T/dt;
 set(ax,'XTickLabel',[0,1,2,3,4])
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:20:100;
 set(ax,'YTickLabel',[0,100,200,300,400,500])
%  ax.ZTick = 8:1:12;
%  set(ax,'ZTickLabel',[8,9,10,11,12],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 axis([0, N, 0, M, 8, 16]);
 plot3(0*ones(1,length(x1)),x1,hv(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1:N-1;
 plot3(x2,M*ones(1,numel(x2)),hv(M,:),'-r','LineWidth',4);

 
obs_data2 = obs_data(:, :, 1:10:1201, :);
sys_data2 = sys_data(:, :, 1:10:1201, :);