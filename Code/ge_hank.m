% Simple HANK model
% Finite difference implicit updating with Poisson income
% includes labor supply
% includes steady state calibration alongside equilibrium objects
% does decompositions
% compares with rep agent
% Greg Kaplan 2024

clear;
close all;
warning('off','AutoDiff:maxmin');

%% OPTIONS
options.Display                 = 3;
options.MakePlotsSteadyState    = 0;
options.MakePlotsIRF            = 1;
options.UseAutoDiffSteadyState  = 0;
options.UseAutoDiffIRF          = 1;
options.AlgorithmIRF            = 1; % 1 = fsolve:levenberg-marquardt, 2 = fsolve:trust-region, 3 = fsolve:trust-region-dogleg, 4= simple broyden
options.SolveRepAgent           = 1;
options.DecomposeConsumption    = 0;

%% PARAMETERS
% calibration options
param.CalibrateRho              = 1; %calibrates discount rate to match target steady state assets
param.CalibrateLabDisutility    = 1; %calibrates labor disutility to mact

% preferences
param.risk_aver         = 1;
param.rhoguess          = 0.006; %0.006; %quarterly; if not calibrate this is actual
param.frisch            = 0.5;
param.labdisutilguess   = 0.4; %0.4; %0.3; %if not calibrated, this is actual
param.maxhours          = 3; %by setting to 3 can make targethours = 1, and then Eh = 1
param.targethours       = 1/3 * param.maxhours; % relevant if param.CalibrateLabDisutility == 1;

%production
param.elastsub      = 10; %100; %10; %>=100 for perfect competition
param.tfp           = 1;

%price stickiness options
param.priceadjcost  = 500; %1000; %500;

%fiscal policy
param.lumptransfer      = 0.278; %0.078; %0.267;   fraction of quarterly steady state output
param.proptransfer      = -0.30;% -0.10; %-0.30; %transfer proportional to idio productivtity: fraction of quarterly output * z/E(z)
param.targetssdebt      = 4.4;  % multiple of quarterly output, relevant if param.CalibrateRho == 1
param.fiscalrule_phi    = 0.0; %0.1; %0.01; %1; % rule is tax = tax^* + phi*(b-b^*); require phi>r^*. phi->infty: adjustment all through tax (set>=10 for b = b^*); phi-> r: adjust tax very slowly

%monetary policy rules
param.taylor_theta  = 10; %0.5; %10; % >0; d/dt i = -theta(i - istar - phi(pi - pistar) - shock); set >=10 for instantaneous rule: i = istar + phi(pi - pistar) + shock 
param.taylor_phi    = 0.0; %1.1; %1.1; % sensitivity to inflation; 
param.inflationss   = 0.005; % steady state inflation rate

% income risk: discretized N(mu,sigma^2)
param.arrivalrate_y   = 0.25; % on average once per year
param.mu_y            = 1;
param.sd_y            = 0.2;
param.ny              = 5;

% asset grids
param.na          = 50;
param.amax        = 100;   %multiple of aggregate output
param.borrow_lim  = 0;    %multiple of aggregate output
param.agrid_par   = 0.4;    %1 for linear, 0 for L-shaped

%time grid
param.Ttrans    = 100; %75; % quarters for transition 400 Ttrans, 100 N trans, dt_init =0.05 works well for helictopter drop; for permanent shock use Ttrans = 50 and dt larger, like 0.2
param.nT        = 50; % %50; % number time steps for transtion, i.e size of Tgrid;
param.dt_initial   = 4/52; % length of first time step (in quarters) - set to Ttrans/nTfor equally spaced steps, else dt steps will increase at a constant exponential rate;
param.Tplotmax = 60;

% regime change in policy rules
param.RegimeChangeOn        = 1;
param.RegimeChangeTime      = 8; % quarters
param.RegimeExpectations    = 'unknown'; % 'known' or 'unknown'
param.fiscalrule_phi_new    = 0.1;
param.taylor_phi_new        = 1.1;
param.RegimeClusterOn       = 1;
param.RegimeClusterShare    = 0.35; % share of transition points placed before T0 when clustering is active
    
% computation
param.maxiter_hjb   = 100;
param.tol_hjb       = 1.0e-10;
param.delta_hjb     = 1.0e5;
param.mindV         = 1.0e-10; % for numerical stability
param.tol_irf       = 1e-6;
param.maxiter_irf   = 10;
param.niter_hours   = 5; %25; %6; %4; set high when solving with a lot of dampening
param.dampen_hours  = 0.05; %0.95; %weight to put on previous iteration of hours when solving at constraint, set close to 1 when solving with high Frisch

%tfp shock
param.tfpshock_size        = 0.05; %-0.05; 
param.tfpshock_decay       = 0.2; % quarterly decay rate

%nominal rate shock
param.rnomshock_size        = 0.0; %0.00125; %quarterly
param.rnom_decay            = 0.2; %only relevant if param.taylor_theta>=10; if param.taylor_theta<10 (i.e. partial adjustment rule) then reversion dictated by rule, not this parameter

%discount rate shock
param.discountshock_size    = 0.0; %-0.001; %shock to discount rate
param.discountshock_decay   = 0.2; % quarterly decay rate

%transfer shock
param.transfershock_size    = 0.0; %percentage of steady state value
param.transfershock_decay   = 0.2; % quarterly decay rate

%guesses
param.rguess    = 0.005; %1-exp(-param.rhoguess) - 0.0005; % a bit lower than inverse of discount rate
param.labguess  = param.targethours;

%% UTILITY FUNCTIONS

%utility of consumption
if param.risk_aver==1
    param.u = @(c)log(c);
else    
    param.u = @(c)(c.^(1-param.risk_aver)-1)./(1-param.risk_aver);
end    
param.u1 = @(c) c.^(-param.risk_aver);
param.u1inv = @(u) u.^(-1./param.risk_aver);

%disutility of work function
param.v = @(h,chi) chi .* (h.^(1+1./param.frisch)) ./ (1+1./param.frisch);
param.v1 = @(h,chi) chi .* (h.^(1./param.frisch));
param.v1inv = @(v,chi) (v./chi).^param.frisch;

%% SET UP GRIDS

% assets
grids.agrid = linspace(0,1, param.na)';
grids.agrid = grids.agrid.^(1./param.agrid_par);
grids.agrid = param.borrow_lim + (param.amax - param.borrow_lim).* grids.agrid;

% asset grid spacing: for partial derivatives
grids.dagrid  =  diff(grids.agrid);
grids.dagridf = [grids.dagrid; grids.dagrid(param.na-1)];
grids.dagridb = [grids.dagrid(1); grids.dagrid];

% trapezoidal rule: for KFE and moments
grids.adelta          = zeros(param.na,1);
grids.adelta(1)       = 0.5*grids.dagrid(1);
grids.adelta(2:param.na-1)  = 0.5*grids.dagrid(1:param.na-2) + 0.5*grids.dagrid(2:param.na-1);
grids.adelta(param.na)      = 0.5*grids.dagrid(param.na-1);

% income: disretize normal distribution
if param.ny>1
    width = fzero(@(x)discrete_normal(param.ny, param.mu_y, param.sd_y,x),2);
    [temp,grids.ygrid,grids.ydist] = discrete_normal(param.ny, param.mu_y, param.sd_y,width);
else
    grids.ygrid = 1;
    grids.ydist = 1;
end

%rescale to enure mean labor efficiency = 1
grids.ygrid = grids.ygrid ./ sum(grids.ydist.*grids.ygrid);
grids.meanlabeff = sum(grids.ydist.*grids.ygrid);        

%grids on (a,y) space
grids.aagrid = repmat(grids.agrid,1, param.ny);
grids.yygrid = repmat(grids.ygrid', param.na,1);
grids.aydelta = repmat(grids.adelta,1, param.ny);
grids.daagrid = repmat(grids.dagrid,1, param.ny);
grids.daagridf = repmat(grids.dagridf,1,param.ny);
grids.daagridb = repmat(grids.dagridb,1,param.ny);
    

% income continuous time transition matrix
grids.ytrans = -param.arrivalrate_y.*eye(param.ny) + param.arrivalrate_y .* repmat(grids.ydist',param.ny,1);
grids.yytrans = kron(grids.ytrans, speye(param.na));

% time grid
grids = setup_transition_grid(grids,param);

%% SOLVE FOR STEADY STATE EQUILIBRIUM

%construct guess of endogenous objects
xguess = [param.rguess; param.labguess];
if param.CalibrateRho ==1; xguess = [xguess; param.rhoguess]; end
if param.CalibrateLabDisutility ==1; xguess = [xguess; param.labdisutilguess]; end


%define market clearing function
fSS = @(x)market_clearing_steadystate(x,param,grids);

% xguess_AD = myAD(xguess);
% resid_AD = fSS(xguess_AD);
% return

%solve and calibrate
if options.UseAutoDiffSteadyState ==0
    options_fsolve = optimoptions('fsolve','Display','iter','FunctionTolerance',1e-7,'SpecifyObjectiveGradient',false);
    % options_fsolve = optimoptions(options_fsolve,'Algorithm','levenberg-marquardt');

    [x,fval,exitflag,output] = fsolve(fSS,xguess,options_fsolve);   

elseif options.UseAutoDiffSteadyState ==1    
    options_fsolve = optimoptions('fsolve','Display','iter','FunctionTolerance',1e-7,'SpecifyObjectiveGradient',true);
    [x,fval,exitflag,output] = fsolve(@(x)fAD(x,fSS),xguess,options_fsolve);    
end

%evaluate at equm prices
[~,steadystate] = market_clearing_steadystate(x,param,grids);

disp(steadystate.rho)
disp(steadystate.labdisutil)

%% SOLVE FOR REP AGENT STEADY STATE
if options.SolveRepAgent == 1
    
    % set discount rate rho so that steady state r same as HA economy
    ra_param = param;
    ra_param.rho = steadystate.r;
    
    %construct guess of endogenous objects at het agent values
    xguess = [steadystate.labor];
    if ra_param.CalibrateLabDisutility ==1; xguess = [xguess; steadystate.labdisutil]; end

    % [res,ra_steadystate] = ra_market_clearing_steadystate(xguess,ra_param);

    %define market clearing function
    fSS_ra = @(x)ra_market_clearing_steadystate(x,ra_param);
    
    %solve and calibrate
    options_fsolve = optimoptions('fsolve','Display','iter','FunctionTolerance',1e-7,'SpecifyObjectiveGradient',false);
    [x,fval,exitflag,output] = fsolve(fSS_ra,xguess,options_fsolve);   

    %evaluate at equm prices
    [~,ra_steadystate] = ra_market_clearing_steadystate(x,ra_param);

end

%% SOLVE FOR IMPULSE RESPONSE

%set shocks
shocks = setup_transition_shocks(param,grids,steadystate);

[irf,irf_solver] = solve_hank_irf_equilibrium(param,grids,steadystate,shocks,options);
irf.solver = irf_solver;


%% SOLVE FOR REP AGENT IRF
if options.SolveRepAgent == 1
    [ra_irf,ra_irf_solver] = solve_ra_irf_equilibrium(ra_param,grids,ra_steadystate,shocks,options);
    ra_irf.solver = ra_irf_solver;
end

%% DO CONSUMPTION DECOMPOSTION: HET AGENT
if options.DecomposeConsumption == 1
    
    %set up price paths from steady state
    xdecomp.r           = steadystate.r .* ones(param.nT,1);
    xdecomp.realwage    = steadystate.realwage .* ones(param.nT,1);
    xdecomp.rho         = steadystate.rho .* ones(param.nT,1);
    % xdecomp.divgrid     = irf.divgrid;
    % xdecomp.transfergrid = irf.transfergrid;
    for it = 1:param.nT
        xdecomp.divgrid{it} = steadystate.divgrid;
        xdecomp.transfergrid{it} = steadystate.transfergrid;
    end

    % direct interest rate
    xdecomp_r       = xdecomp;
    xdecomp_r.r     = irf.r;    
    decomp_r = market_clearing_decomp(xdecomp_r,param,grids,steadystate);

    % wage
    xdecomp_w           = xdecomp;
    xdecomp_w.realwage  = irf.realwage;    
    decomp_w = market_clearing_decomp(xdecomp_w,param,grids,steadystate);

    % transfer
    xdecomp_T           = xdecomp;
    xdecomp_T.transfergrid  = irf.transfergrid;    
    decomp_T = market_clearing_decomp(xdecomp_T,param,grids,steadystate);

    % dividends
    xdecomp_D           = xdecomp;
    xdecomp_D.divgrid = irf.divgrid;    
    decomp_D = market_clearing_decomp(xdecomp_D,param,grids,steadystate);

    % consumption
    figure(4);
    hold on;
    plot(grids.Tgrid,irf.Ec./steadystate.Ec,'k-','LineWidth',3);
    plot(grids.Tgrid,decomp_r.Ec./steadystate.Ec,'b-','LineWidth',2);
    plot(grids.Tgrid,decomp_w.Ec./steadystate.Ec,'r-.','LineWidth',2);
    plot(grids.Tgrid,decomp_T.Ec./steadystate.Ec,'m-','LineWidth',2);
    plot(grids.Tgrid,decomp_D.Ec./steadystate.Ec,'g-','LineWidth',2);
    plot(grids.Tgrid,ones(param.nT,1),'k-','LineWidth',1);
    hold off;
    grid on;
    legend('Total Response','Real Rate','Real Wage','Govt Transfers','Monopoly Profits','Location','best','FontSize',14)
    xlim([0 param.Tplotmax]);
    title('Consumption: Het Agent');

    
end  

%% %% DO CONSUMPTION DECOMPOSTION: REP AGENT

if options.DecomposeConsumption == 1 && options.SolveRepAgent == 1
    %set up price paths from steady state
    ra_xdecomp.r           = ra_steadystate.r .* ones(param.nT,1);
    ra_xdecomp.realwage    = ra_steadystate.realwage .* ones(param.nT,1);
    ra_xdecomp.rho         = ra_steadystate.rho .* ones(param.nT,1);
    ra_xdecomp.dividend     = ra_steadystate.dividend.* ones(param.nT,1);
    ra_xdecomp.transfer     = ra_steadystate.transfer.* ones(param.nT,1);
    

    % direct interest rate
    ra_xdecomp_r       = ra_xdecomp;
    ra_xdecomp_r.r     = ra_irf.r;    
    ra_decomp_r = ra_market_clearing_decomp(ra_xdecomp_r,ra_param,grids,ra_steadystate);

    % wage
    ra_xdecomp_w           = ra_xdecomp;
    ra_xdecomp_w.realwage  = ra_irf.realwage;    
    ra_decomp_w = ra_market_clearing_decomp(ra_xdecomp_w,ra_param,grids,ra_steadystate);

    % transfer
    ra_xdecomp_T           = ra_xdecomp;
    ra_xdecomp_T.transfer  = ra_irf.transfer;    
    ra_decomp_T = ra_market_clearing_decomp(ra_xdecomp_T,ra_param,grids,ra_steadystate);

    % dividends
    ra_xdecomp_D           = ra_xdecomp;
    ra_xdecomp_D.dividend = ra_irf.dividend;    
    ra_decomp_D = ra_market_clearing_decomp(ra_xdecomp_D,ra_param,grids,ra_steadystate);

    % consumption
    figure(5);
    hold on;
    plot(grids.Tgrid,ra_irf.Ec./ra_steadystate.Ec,'k-','LineWidth',3);
    plot(grids.Tgrid,ra_decomp_r.Ec./ra_steadystate.Ec,'b-','LineWidth',2);
    plot(grids.Tgrid,ra_decomp_w.Ec./ra_steadystate.Ec,'r-.','LineWidth',2);
    plot(grids.Tgrid,ra_decomp_T.Ec./ra_steadystate.Ec,'m-','LineWidth',2);
    plot(grids.Tgrid,ra_decomp_D.Ec./ra_steadystate.Ec,'g-','LineWidth',2);
    plot(grids.Tgrid,ones(ra_param.nT,1),'k-','LineWidth',1);
    hold off;
    grid on;
    legend('Total Response','Real Rate','Real Wage','Govt Transfers','Monopoly Profits','Location','best','FontSize',14)
    xlim([0 param.Tplotmax]);
    title('Consumption: Rep Agent');

end

%% MAKE PLOTS
if options.MakePlotsSteadyState ==1 
    PlotResultsSteadyState(1,param,grids,steadystate)
end

if options.MakePlotsIRF==1 
    PlotResultsIRF(2,param,grids,steadystate,irf)
end

if options.SolveRepAgent ==1
    PlotResultsCompareRA(3,grids,param,ra_param,steadystate,ra_steadystate,irf, ra_irf)
end
%% FUNCTIONS


%%
function [res,steadystate] = market_clearing_steadystate(x,param,grids)
     
    %extract some parameters and grids
    adelta = grids.adelta;
    yygrid = grids.yygrid;
    ydist = grids.ydist;
    na = param.na;
    ny = param.ny;
    agrid = grids.agrid;
    aagrid = grids.aagrid;
    aydelta = grids.aydelta;
    
    %extract input variables depending on configuration
    ii = 1;

    % eqm real rate
    r = x(ii);    
    ii = ii +1;    
    rnom = r + param.inflationss;
    
    % eqm labor input    
    labor = x(ii);
    ii = ii +1;

    % discount rate if calibrating to debt target
    if param.CalibrateRho  ==1
        rho = x(ii);
        ii = ii +1;
    else
        rho = param.rhoguess;
    end

    % labor disutility if calibrating to hours target    
    if param.CalibrateLabDisutility ==1
        labdisutil = x(ii);
        ii = ii +1;
    else
        labdisutil = param.labdisutilguess;
    end               

    %production and wages
    if param.elastsub >= 100; markup = 1; end
    if param.elastsub < 100; markup = param.elastsub/(param.elastsub-1); end
    realmc = 1/markup;
    tfp = param.tfp;
    realoutput = tfp*labor;
    realwage = realoutput * realmc / labor;        
    
    %profits and dividends    
    realprofit = (1- realmc)*realoutput;
    divgrid = (yygrid./grids.meanlabeff) .* realprofit;
%     divgrid = ones(na,ny).* realprofit;
    %taxes and transfers
    transfergrid = (param.lumptransfer .* ones(na,ny) + param.proptransfer.*(yygrid./grids.meanlabeff)).*realoutput;

    % initialize value function
%     Vguess = zeros(na,ny) + 0.*r;
    Vguess = ( param.u(r.*aagrid + realwage.*yygrid + transfergrid + divgrid)....
                    -param.v(param.targethours, labdisutil)) ./ rho;

    % solve HJB
    V = Vguess;
    
    Vdiff = 1;
    iter = 0;

    while iter <= param.maxiter_hjb && Vdiff>param.tol_hjb
        iter = iter + 1;
        
        %update value function
        passedvars.r = r;
        passedvars.realwage = realwage;
        passedvars.delta = param.delta_hjb;
        passedvars.divgrid = divgrid;
        passedvars.transfergrid = transfergrid;
        passedvars.labdisutil = labdisutil;
        passedvars.rho = rho;
        [Vnew,A,con,sav,hour] = UpdateHJB(V,param,grids,passedvars);

        Vdiff = max(abs(Vnew(:)-V(:)));
        V = Vnew;
    end 


    % solve KFE
    lgvecadj = [A' ones(na*ny,1); ones(1,na*ny) 0] \ [zeros(na*ny,1); 1];
    gvecadj = lgvecadj(1:na*ny);
    gmatadj = reshape(gvecadj,na,ny);
    gmat    = gmatadj./aydelta;

 
    %asset supply
    gamarg = sum(gmat,2);
    Ea = sum(gamarg.*agrid.*adelta);
    Ec = sum(con(:).*gmat(:).*aydelta(:));
    Eh = sum(hour(:).*gmat(:).*aydelta(:));
    Eefflab = sum(hour(:).*yygrid(:).*gmat(:).*aydelta(:));

    %real assets of government
    steadystate.realsurplus =  - param.lumptransfer*realoutput - param.proptransfer*realoutput;
    steadystate.realgovdebt = steadystate.realsurplus/r;
    

    %residuals
    res_assetmarket = Ea - steadystate.realgovdebt;
    res_labormarket = Eefflab - labor;
    res_debttarget = Ea - param.targetssdebt*realoutput;
    res_hourtarget = Eh - param.targethours;
    
    res = [res_assetmarket; res_labormarket];
    if param.CalibrateRho ==1; res = [res; res_debttarget]; end    
    if param.CalibrateLabDisutility ==1; res = [res; res_hourtarget]; end
    


    %package output
    if nargout>1   
        steadystate.r = r;
        steadystate.rnom = rnom;
        steadystate.inflation = param.inflationss;
        steadystate.labor = labor;
        steadystate.rho = rho;
        steadystate.labdisutil = labdisutil;
        steadystate.realwage = realwage;   
        steadystate.realmc = realmc;
        steadystate.con = con;
        steadystate.sav = sav;
        steadystate.hour = hour;
        steadystate.gamarg = gamarg;
        steadystate.gmat = gmat;
        steadystate.A = A;
        steadystate.V = V;
        steadystate.tfp = tfp;
        steadystate.realoutput = realoutput;
        steadystate.Ea = Ea;
        steadystate.Ec = Ec;
        steadystate.Eh = Eh;
        steadystate.Eefflab = Eefflab;
        steadystate.divgrid = divgrid;
        steadystate.transfergrid = transfergrid;
    end
end

%%
function [res,steadystate] = ra_market_clearing_steadystate(x,param)
     
    
    %extract input variables depending on configuration
    ii = 1;
    
    % eqm labor input    
    labor = x(ii);
    ii = ii +1;
    
    % labor disutility if calibrating to hours target    
    if param.CalibrateLabDisutility ==1
        labdisutil = x(ii);
        ii = ii +1;
    else
        labdisutil = param.labdisutilguess;
    end               

    % eqm real rate
    r = param.rho;
    rnom = r + param.inflationss;

    %production and wages
    if param.elastsub >= 100; markup = 1; end
    if param.elastsub < 100; markup = param.elastsub/(param.elastsub-1); end
    realmc = 1/markup;
    tfp = param.tfp;
    realoutput = tfp*labor;
    realwage = realoutput * realmc / labor;        
    
    %profits and dividends    
    realprofit = (1- realmc)*realoutput;
    dividend = realprofit;
    
    %taxes and transfers
    transfer = (param.lumptransfer + param.proptransfer).*realoutput;

    %real assets of government
    realsurplus =  - transfer;
    realgovdebt = realsurplus/r;

    %consumption from budget constraint
    Ea = realgovdebt;
    Ec  = r.*Ea + realwage*labor + transfer + dividend;
    
    %hours from FOC
    Eh = param.v1inv(param.u1(Ec) .* realwage, labdisutil );
    Eefflab  = Eh;    
    
    %residuals    
    res_labormarket = Eefflab - labor;
    res_hourtarget = Eh - param.targethours;
        
    
    res = [res_labormarket];
    if param.CalibrateLabDisutility ==1; res = [res; res_hourtarget]; end
    


    %package output
    if nargout>1   
        steadystate.r = r;
        steadystate.rnom = rnom;
        steadystate.inflation = param.inflationss;
        steadystate.labor = labor;
        steadystate.rho = param.rho;
        steadystate.labdisutil = labdisutil;
        steadystate.realwage = realwage;   
        steadystate.realmc = realmc;
        steadystate.realgovdebt = realgovdebt;
        steadystate.realsurplus = realsurplus;
        steadystate.tfp = tfp;
        steadystate.realoutput = realoutput;
        steadystate.Ea = Ea;
        steadystate.Ec = Ec;
        steadystate.Eh = Eh;
        steadystate.Eefflab = Eefflab;
        steadystate.dividend = dividend;
        steadystate.transfer = transfer;
    end
end

%%
function [irf,solver] = solve_hank_irf_equilibrium(param,grids,steadystate,shocks,options)

    regime_on = grids.regime_on;
    scenario = lower(param.RegimeExpectations);
    xguess = make_irf_initial_guess(grids.nT);

    if regime_on && strcmp(scenario,'unknown')
        policy_old = constant_policy_sequences(param,grids.nT,0);
        init_old = default_hank_initial_state(steadystate,0);
        objective_old = @(x)market_clearing_irf_segment(x,param,grids,steadystate,shocks,policy_old,init_old);
        [x_old,old_irf,solver_old] = solve_transition_equilibrium(objective_old,xguess,param,options);

        tail_grids = slice_transition_grid(grids,grids.regime_idx);
        tail_shocks = slice_transition_shocks(shocks,grids.regime_idx);
        tail_policy = constant_policy_sequences(param,tail_grids.nT,1);
        tail_init = hank_initial_state_from_irf(old_irf,grids.regime_idx);
        tail_init.force_inherited_debt = 0;
        xguess_tail = make_irf_guess_from_path(old_irf,steadystate,grids.regime_idx);
        target_bT0 = old_irf.realgovdebt(grids.regime_idx);
        objective_tail = @(x)market_clearing_irf_unknown_tail(x,param,tail_grids,steadystate,tail_shocks,tail_policy,tail_init,target_bT0);
        [x_tail,tail_irf,solver_tail] = solve_transition_equilibrium(objective_tail,xguess_tail,param,options);

        irf = splice_transition_structs(old_irf,tail_irf,grids.regime_idx);
        irf.pre_switch = old_irf;
        irf.post_switch = tail_irf;
        irf.regime_expectations = 'unknown';
        irf.regime_idx = grids.regime_idx;
        irf.regime_time = grids.regime_time;

        solver.old = solver_old;
        solver.old.x = x_old;
        solver.tail = solver_tail;
        solver.tail.x = x_tail;
        solver.scenario = 'unknown';
    else
        policy = build_policy_sequences(param,grids);
        init_state = default_hank_initial_state(steadystate,0);
        objective = @(x)market_clearing_irf_segment(x,param,grids,steadystate,shocks,policy,init_state);
        [x_irf,irf,solver] = solve_transition_equilibrium(objective,xguess,param,options);
        irf.regime_expectations = ternary(regime_on,'known','none');
        irf.regime_idx = grids.regime_idx;
        irf.regime_time = grids.regime_time;
        solver.x = x_irf;
        solver.scenario = irf.regime_expectations;
    end
end

%%
function [irf,solver] = solve_ra_irf_equilibrium(param,grids,steadystate,shocks,options)

    regime_on = grids.regime_on;
    scenario = lower(param.RegimeExpectations);
    xguess = make_irf_initial_guess(grids.nT);

    if regime_on && strcmp(scenario,'unknown')
        policy_old = constant_policy_sequences(param,grids.nT,0);
        init_old = default_ra_initial_state(steadystate,0);
        objective_old = @(x)ra_market_clearing_irf_segment(x,param,grids,steadystate,shocks,policy_old,init_old);
        [x_old,old_irf,solver_old] = solve_transition_equilibrium(objective_old,xguess,param,options);

        tail_grids = slice_transition_grid(grids,grids.regime_idx);
        tail_shocks = slice_transition_shocks(shocks,grids.regime_idx);
        tail_policy = constant_policy_sequences(param,tail_grids.nT,1);
        tail_init = ra_initial_state_from_irf(old_irf,grids.regime_idx);
        tail_init.force_inherited_debt = 0;
        xguess_tail = make_irf_guess_from_path(old_irf,steadystate,grids.regime_idx);
        target_bT0 = old_irf.realgovdebt(grids.regime_idx);
        objective_tail = @(x)ra_market_clearing_irf_unknown_tail(x,param,tail_grids,steadystate,tail_shocks,tail_policy,tail_init,target_bT0);
        [x_tail,tail_irf,solver_tail] = solve_transition_equilibrium(objective_tail,xguess_tail,param,options);

        irf = splice_transition_structs(old_irf,tail_irf,grids.regime_idx);
        irf.pre_switch = old_irf;
        irf.post_switch = tail_irf;
        irf.regime_expectations = 'unknown';
        irf.regime_idx = grids.regime_idx;
        irf.regime_time = grids.regime_time;

        solver.old = solver_old;
        solver.old.x = x_old;
        solver.tail = solver_tail;
        solver.tail.x = x_tail;
        solver.scenario = 'unknown';
    else
        policy = build_policy_sequences(param,grids);
        init_state = default_ra_initial_state(steadystate,0);
        objective = @(x)ra_market_clearing_irf_segment(x,param,grids,steadystate,shocks,policy,init_state);
        [x_irf,irf,solver] = solve_transition_equilibrium(objective,xguess,param,options);
        irf.regime_expectations = ternary(regime_on,'known','none');
        irf.regime_idx = grids.regime_idx;
        irf.regime_time = grids.regime_time;
        solver.x = x_irf;
        solver.scenario = irf.regime_expectations;
    end
end

%%
function [xsol,irf,solver] = solve_transition_equilibrium(objective,xguess,param,options)

    sample_residual = objective(xguess);
    n_unknowns = numel(xguess);
    n_residuals = numel(sample_residual);
    nonsquare_system = n_residuals ~= n_unknowns;

    if options.AlgorithmIRF < 4
        options_fsolve = optimoptions('fsolve','Display','iter','FunctionTolerance',param.tol_irf,'MaxIterations',param.maxiter_irf);
        if nonsquare_system || options.AlgorithmIRF==1
            options_fsolve = optimoptions(options_fsolve,'Algorithm','levenberg-marquardt');
        elseif options.AlgorithmIRF==2
            options_fsolve = optimoptions(options_fsolve,'Algorithm','trust-region-dogleg');
        elseif options.AlgorithmIRF==3
            options_fsolve = optimoptions(options_fsolve,'Algorithm','trust-region');
        end

        if options.UseAutoDiffIRF == 0 || nonsquare_system
            options_fsolve = optimoptions(options_fsolve,'SpecifyObjectiveGradient',false);
            [xsol,fval,exitflag,output] = fsolve(objective,xguess,options_fsolve);
        else
            options_fsolve = optimoptions(options_fsolve,'SpecifyObjectiveGradient',true);
            [xsol,fval,exitflag,output] = fsolve(@(x)fAD(x,objective),xguess,options_fsolve);
        end
    else
        if nonsquare_system
            error('Unknown-regime tail solve adds a debt-matching condition, so simple Broyden is not supported. Use AlgorithmIRF = 1.');
        end
        disp(' Computing Jacobian at initial guess')
        xguess_AD = myAD(xguess);
        resid_AD = objective(xguess_AD);
        initjac = getderivs(resid_AD);
        disp(' Solving using simple Broyden Method')

        xsol = fn_simple_broyden(objective,xguess,inv(initjac),20,param.tol_irf,2);
        fval = objective(xsol);
        exitflag = NaN;
        output = struct();
    end

    [residuals,irf] = objective(xsol);
    solver.fval = fval;
    solver.exitflag = exitflag;
    solver.output = output;
    solver.residuals = residuals;
end

%%
function [res,irf] = market_clearing_irf_unknown_tail(x,param,grids,steadystate,shocks,policy,initial_state,target_bT0)
    [base_res,irf] = market_clearing_irf_segment(x,param,grids,steadystate,shocks,policy,initial_state);
    irf.res_debtmatch = irf.realgovdebt(1) - target_bT0;
    res = [base_res; irf.res_debtmatch];
end

%%
function [res,irf] = ra_market_clearing_irf_unknown_tail(x,param,grids,steadystate,shocks,policy,initial_state,target_bT0)
    [base_res,irf] = ra_market_clearing_irf_segment(x,param,grids,steadystate,shocks,policy,initial_state);
    irf.res_debtmatch = irf.realgovdebt(1) - target_bT0;
    res = [base_res; irf.res_debtmatch];
end

%%
function [res,irf] = market_clearing_irf_segment(x,param,grids,steadystate,shocks,policy,initial_state)

    na = param.na;
    ny = param.ny;
    nT = grids.nT;
    yygrid = grids.yygrid;

    irf.inflation = steadystate.inflation + [x(1:nT-1); 0];
    irf.inflationdot = [diff(irf.inflation); steadystate.inflation - irf.inflation(nT)] ./ grids.dt_trans;
    irf.inflationrelss = irf.inflation - steadystate.inflation;

    if param.taylor_theta >= 10
        irf.rnom = steadystate.rnom + policy.taylor_phi .* irf.inflationrelss + shocks.rnomshock;
    else
        irf.rnom = zeros(nT,1) + 0.*x(1);
        irf.rnom(1) = initial_state.rnom + shocks.rnomshock(1);
        for it = 1:nT-1
            target = steadystate.rnom + policy.taylor_phi(it) .* irf.inflationrelss(it) + shocks.rnomshock(it);
            irf.rnom(it+1) = (irf.rnom(it) + grids.dt_trans(it) * param.taylor_theta * target) ...
                             / (1 + grids.dt_trans(it) * param.taylor_theta);
        end
    end

    irf.r = irf.rnom - irf.inflation;

    irf.tfp = steadystate.tfp + shocks.tfpshock;
    irf.ydot = [x(nT:2*nT-2); 0];
    irf.realoutput = zeros(nT,1) + 0.*x(1);
    irf.realoutput(nT) = steadystate.realoutput;
    for it = nT-1:-1:1
        irf.realoutput(it) = irf.realoutput(it+1) - grids.dt_trans(it) .* irf.ydot(it);
    end

    irf.rho = steadystate.rho + shocks.discountshock;
    irf.realmc = ((irf.r - irf.ydot ./ irf.realoutput) .* irf.inflationrelss - irf.inflationdot) ...
                 * (param.priceadjcost / param.elastsub) + steadystate.realmc;
    irf.markup = 1 ./ irf.realmc;
    irf.priceadj = (param.priceadjcost / 2) .* (irf.inflationrelss .^ 2) .* irf.realoutput;

    irf.labor = irf.realoutput ./ irf.tfp;
    irf.realprofit = (1 - irf.realmc) .* irf.realoutput - irf.priceadj;
    irf.realwage = irf.realoutput .* irf.realmc ./ irf.labor;

    for it = 1:nT
        irf.divgrid{it} = (yygrid ./ grids.meanlabeff) .* irf.realprofit(it);
    end

    [irf.realgovdebt,irf.realsurplus] = solve_government_path(param,grids,steadystate,shocks,policy,irf.r,initial_state.realgovdebt,initial_state.force_inherited_debt);
    irf.lumptransfer = param.lumptransfer - (irf.realsurplus - steadystate.realsurplus) ./ steadystate.realoutput;
    irf.transfer = (irf.lumptransfer + param.proptransfer) .* steadystate.realoutput + shocks.transfershock;

    for it = 1:nT
        irf.transfergrid{it} = (irf.lumptransfer(it) .* ones(na,ny) + param.proptransfer .* (yygrid ./ grids.meanlabeff)) .* steadystate.realoutput;
        irf.transfergrid{it} = irf.transfergrid{it} + shocks.transfershock(it);
    end

    irf.V = cell(nT,1);
    irf.A = cell(nT,1);
    irf.con = cell(nT,1);
    irf.sav = cell(nT,1);
    irf.hour = cell(nT,1);

    for it = nT:-1:1
        passedvars.r = irf.r(it);
        passedvars.realwage = irf.realwage(it);
        passedvars.delta = grids.dt_trans(it);
        passedvars.divgrid = irf.divgrid{it};
        passedvars.transfergrid = irf.transfergrid{it};
        passedvars.labdisutil = steadystate.labdisutil;
        passedvars.rho = irf.rho(it);

        if it == nT
            Vnext = steadystate.V;
        else
            Vnext = irf.V{it+1};
        end

        [irf.V{it},irf.A{it},irf.con{it},irf.sav{it},irf.hour{it}] = UpdateHJB(Vnext,param,grids,passedvars);
    end

    irf.gmat = cell(nT,1);
    irf.gvecadj = cell(nT,1);
    irf.Ea = zeros(nT,1) + 0.*x(1);
    irf.Eh = zeros(nT,1) + 0.*x(1);
    irf.Ec = zeros(nT,1) + 0.*x(1);
    irf.Eefflab = zeros(nT,1) + 0.*x(1);

    irf.gmat{1} = initial_state.gmat;
    irf.gvecadj{1} = reshape(grids.aydelta .* irf.gmat{1},na*ny,1);

    for it = 1:nT-1
        irf.gvecadj{it+1} = (speye(na*ny) - irf.A{it}' * grids.dt_trans(it)) \ irf.gvecadj{it};
        irf.gmat{it+1} = reshape(irf.gvecadj{it+1},na,ny) ./ grids.aydelta;
    end

    for it = 1:nT
        irf.gamarg{it} = sum(irf.gmat{it},2);
        irf.Ea(it) = sum(irf.gamarg{it} .* grids.agrid .* grids.adelta);
        irf.Eh(it) = sum(irf.hour{it}(:) .* irf.gmat{it}(:) .* grids.aydelta(:));
        irf.Ec(it) = sum(irf.con{it}(:) .* irf.gmat{it}(:) .* grids.aydelta(:));
        irf.Eefflab(it) = sum(irf.hour{it}(:) .* yygrid(:) .* irf.gmat{it}(:) .* grids.aydelta(:));
    end

    irf.res_assetmarket = irf.Ea(2:nT) - irf.realgovdebt(2:nT);
    irf.res_labormarket = irf.Eefflab - irf.labor;
    irf.policy_taylor_phi = policy.taylor_phi;
    irf.policy_fiscal_phi = policy.fiscal_phi;
    res = [irf.res_assetmarket; irf.res_labormarket];
end

%%
function [res,irf] = ra_market_clearing_irf_segment(x,param,grids,steadystate,shocks,policy,initial_state)

    nT = grids.nT;

    irf.inflation = steadystate.inflation + [x(1:nT-1); 0];
    irf.inflationdot = [diff(irf.inflation); steadystate.inflation - irf.inflation(nT)] ./ grids.dt_trans;
    irf.inflationrelss = irf.inflation - steadystate.inflation;

    if param.taylor_theta >= 10
        irf.rnom = steadystate.rnom + policy.taylor_phi .* irf.inflationrelss + shocks.rnomshock;
    else
        irf.rnom = zeros(nT,1) + 0.*x(1);
        irf.rnom(1) = initial_state.rnom + shocks.rnomshock(1);
        for it = 1:nT-1
            target = steadystate.rnom + policy.taylor_phi(it) .* irf.inflationrelss(it) + shocks.rnomshock(it);
            irf.rnom(it+1) = (irf.rnom(it) + grids.dt_trans(it) * param.taylor_theta * target) ...
                             / (1 + grids.dt_trans(it) * param.taylor_theta);
        end
    end

    irf.r = irf.rnom - irf.inflation;

    irf.tfp = steadystate.tfp + shocks.tfpshock;
    irf.ydot = [x(nT:2*nT-2); 0];
    irf.realoutput = zeros(nT,1) + 0.*x(1);
    irf.realoutput(nT) = steadystate.realoutput;
    for it = nT-1:-1:1
        irf.realoutput(it) = irf.realoutput(it+1) - grids.dt_trans(it) .* irf.ydot(it);
    end

    irf.rho = steadystate.rho + shocks.discountshock;
    irf.realmc = ((irf.r - irf.ydot ./ irf.realoutput) .* irf.inflationrelss - irf.inflationdot) ...
                 * (param.priceadjcost / param.elastsub) + steadystate.realmc;
    irf.markup = 1 ./ irf.realmc;
    irf.priceadj = (param.priceadjcost / 2) .* (irf.inflationrelss .^ 2) .* irf.realoutput;

    irf.labor = irf.realoutput ./ irf.tfp;
    irf.realprofit = (1 - irf.realmc) .* irf.realoutput - irf.priceadj;
    irf.dividend = irf.realprofit;
    irf.realwage = irf.realoutput .* irf.realmc ./ irf.labor;

    [irf.realgovdebt,irf.realsurplus] = solve_government_path(param,grids,steadystate,shocks,policy,irf.r,initial_state.realgovdebt,initial_state.force_inherited_debt);
    irf.lumptransfer = param.lumptransfer - (irf.realsurplus - steadystate.realsurplus) ./ steadystate.realoutput;
    irf.transfer = (irf.lumptransfer + param.proptransfer) .* steadystate.realoutput + shocks.transfershock;

    u1c = zeros(nT,1) + 0.*x(1);
    irf.Ec = zeros(nT,1) + 0.*x(1);
    u1c(nT) = param.u1(steadystate.Ec);
    irf.Ec(nT) = param.u1inv(u1c(nT));
    for it = nT-1:-1:1
        u1c(it) = u1c(it+1) ./ (1 + (irf.rho(it) - irf.r(it)) .* grids.dt_trans(it));
        irf.Ec(it) = param.u1inv(u1c(it));
    end

    irf.Eh = zeros(nT,1) + 0.*x(1);
    for it = 1:nT
        lu1 = param.u1(irf.Ec(it));
        irf.Eh(it) = param.v1inv(lu1 .* irf.realwage(it), steadystate.labdisutil);
    end
    irf.Eefflab = irf.Eh;

    irf.Ea = zeros(nT,1) + 0.*x(1);
    irf.Ea(1) = initial_state.Ea;
    irf.lflow = zeros(nT-1,1) + 0.*x(1);
    for it = 1:nT-1
        irf.lflow(it) = irf.Eh(it) .* irf.realwage(it) + irf.transfer(it) + irf.dividend(it) - irf.Ec(it);
        irf.Ea(it+1) = (irf.Ea(it) + grids.dt_trans(it) .* irf.lflow(it)) ./ (1 - grids.dt_trans(it) .* irf.r(it));
    end

    irf.res_assetmarket = irf.Ea(2:nT) - irf.realgovdebt(2:nT);
    irf.res_labormarket = irf.Eefflab - irf.labor;
    irf.policy_taylor_phi = policy.taylor_phi;
    irf.policy_fiscal_phi = policy.fiscal_phi;
    res = [irf.res_assetmarket; irf.res_labormarket];
end

%%
function shocks = setup_transition_shocks(param,grids,steadystate)

    % Use absolute calendar time at the transition grid points so the
    % exogenous process is invariant to how the grid is split around T0.
    tshock = grids.Tgrid;
    shocks.time = tshock;
    shocks.tfpshock = steadystate.tfp * param.tfpshock_size .* exp(-param.tfpshock_decay .* tshock);
    shocks.discountshock = param.discountshock_size .* exp(-param.discountshock_decay .* tshock);
    shocks.rnomshock = param.rnomshock_size .* exp(-param.rnom_decay .* tshock);
    shocks.transfershock = param.lumptransfer * steadystate.realoutput * param.transfershock_size .* exp(-param.transfershock_decay .* tshock);
end

%%
function grids = setup_transition_grid(grids,param)

    regime_on = isfield(param,'RegimeChangeOn') && param.RegimeChangeOn == 1 ...
                && param.RegimeChangeTime > 0 && param.RegimeChangeTime < param.Ttrans;

    if regime_on
        nT_pre_baseline = max(1, min(param.nT-1, round(param.nT * param.RegimeChangeTime / param.Ttrans)));
        if isfield(param,'RegimeClusterOn') && param.RegimeClusterOn == 1
            nT_pre_cluster = max(1, min(param.nT-1, round(param.nT * param.RegimeClusterShare)));
            nT_pre = max(nT_pre_baseline,nT_pre_cluster);
        else
            nT_pre = nT_pre_baseline;
        end
        nT_post = param.nT - nT_pre;
        dt_pre = make_transition_steps(param.RegimeChangeTime,nT_pre,param.dt_initial);
        dt_post = make_transition_steps(param.Ttrans - param.RegimeChangeTime,nT_post,param.dt_initial);
        grids.dt_trans = [dt_pre; dt_post];
        grids.regime_idx = nT_pre + 1;
        grids.regime_time = param.RegimeChangeTime;
    else
        grids.dt_trans = make_transition_steps(param.Ttrans,param.nT,param.dt_initial);
        grids.regime_idx = [];
        grids.regime_time = [];
    end

    grids.nT = numel(grids.dt_trans);
    grids.Tstart = [0; cumsum(grids.dt_trans(1:end-1))];
    grids.Tgrid = cumsum(grids.dt_trans);
    grids.regime_on = regime_on;
end

%%
function dt = make_transition_steps(Ttotal,nT,dt_initial)

    if nT <= 0
        dt = zeros(0,1);
        return
    elseif nT == 1
        dt = Ttotal;
        return
    end

    avg_dt = Ttotal / nT;
    if abs(dt_initial - avg_dt) < 1.0e-12 || dt_initial >= avg_dt
        dt = avg_dt .* ones(nT,1);
    else
        f = @(x) Ttotal .* (x - 1) ./ (x .^ nT - 1) - dt_initial;
        try
            dt_growth = fzero(f,1.2);
            dt = dt_initial .* dt_growth .^ (0:nT-1)';
        catch
            dt = avg_dt .* ones(nT,1);
        end
    end
    dt = dt .* (Ttotal ./ sum(dt));
end

%%
function xguess = make_irf_initial_guess(nT)
    xguess = zeros(2*nT-1,1);
end

%%
function xguess = make_irf_guess_from_path(irf,steadystate,start_idx)
    pidev = irf.inflation(start_idx:end-1) - steadystate.inflation;
    ydev = irf.ydot(start_idx:end);
    xguess = [pidev; ydev];
end

%%
function policy = build_policy_sequences(param,grids)
    policy = constant_policy_sequences(param,grids.nT,0);
    if grids.regime_on
        policy.taylor_phi(grids.regime_idx:end) = param.taylor_phi_new;
        policy.fiscal_phi(grids.regime_idx:end) = param.fiscalrule_phi_new;
    end
    policy.use_terminal_regime_for_debt = 1;
    policy.terminal_taylor_phi = policy.taylor_phi(end);
end

%%
function policy = constant_policy_sequences(param,nT,use_new)
    policy.taylor_phi = param.taylor_phi .* ones(nT,1);
    policy.fiscal_phi = param.fiscalrule_phi .* ones(nT,1);
    if use_new == 1
        policy.taylor_phi(:) = param.taylor_phi_new;
        policy.fiscal_phi(:) = param.fiscalrule_phi_new;
    end
    policy.use_terminal_regime_for_debt = 0;
    policy.terminal_taylor_phi = policy.taylor_phi(end);
end

%%
function initial_state = default_hank_initial_state(steadystate,force_inherited_debt)
    initial_state.gmat = steadystate.gmat;
    initial_state.realgovdebt = steadystate.realgovdebt;
    initial_state.rnom = steadystate.rnom;
    initial_state.force_inherited_debt = force_inherited_debt;
end

%%
function initial_state = hank_initial_state_from_irf(irf,idx)
    initial_state.gmat = irf.gmat{idx};
    initial_state.realgovdebt = irf.realgovdebt(idx);
    initial_state.rnom = irf.rnom(idx);
    initial_state.force_inherited_debt = 1;
end

%%
function initial_state = default_ra_initial_state(steadystate,force_inherited_debt)
    initial_state.Ea = steadystate.Ea;
    initial_state.realgovdebt = steadystate.realgovdebt;
    initial_state.rnom = steadystate.rnom;
    initial_state.force_inherited_debt = force_inherited_debt;
end

%%
function initial_state = ra_initial_state_from_irf(irf,idx)
    initial_state.Ea = irf.Ea(idx);
    initial_state.realgovdebt = irf.realgovdebt(idx);
    initial_state.rnom = irf.rnom(idx);
    initial_state.force_inherited_debt = 1;
end

%%
function tail_grids = slice_transition_grid(grids,start_idx)
    tail_grids = grids;
    tail_grids.dt_trans = grids.dt_trans(start_idx:end);
    tail_grids.nT = numel(tail_grids.dt_trans);
    tail_grids.Tstart = [0; cumsum(tail_grids.dt_trans(1:end-1))];
    tail_grids.Tgrid = cumsum(tail_grids.dt_trans);
    tail_grids.regime_on = 0;
    tail_grids.regime_idx = [];
    tail_grids.regime_time = [];
end

%%
function tail_shocks = slice_transition_shocks(shocks,start_idx)
    shockfields = fieldnames(shocks);
    for ifield = 1:numel(shockfields)
        tail_shocks.(shockfields{ifield}) = shocks.(shockfields{ifield})(start_idx:end);
    end
end

%%
function [realgovdebt,realsurplus] = solve_government_path(param,grids,steadystate,shocks,policy,r,initial_debt,force_inherited_debt)

    nT = grids.nT;
    realgovdebt = zeros(nT,1) + 0.*r(1);

    if all(policy.fiscal_phi >= 10)
        realgovdebt(:) = initial_debt;
        realsurplus = r .* realgovdebt;
        return
    end

    if isfield(policy,'use_terminal_regime_for_debt') && policy.use_terminal_regime_for_debt == 1
        use_forward = force_inherited_debt || policy.terminal_taylor_phi >= 3;
    else
        use_forward = force_inherited_debt || all(policy.taylor_phi >= 3);
    end
    if use_forward
        realgovdebt(1) = initial_debt;
        for it = 1:nT-1
            fiscal_phi = policy.fiscal_phi(it);
            realgovdebt(it+1) = (realgovdebt(it) * (1 - fiscal_phi * grids.dt_trans(it)) ...
                                - grids.dt_trans(it) * (steadystate.r - fiscal_phi) * steadystate.realgovdebt ...
                                + grids.dt_trans(it) * shocks.transfershock(it)) ...
                                / (1 - grids.dt_trans(it) * r(it));
        end
    else
        realgovdebt(nT) = steadystate.realgovdebt;
        for it = nT-1:-1:1
            fiscal_phi = policy.fiscal_phi(it);
            realgovdebt(it) = (realgovdebt(it+1) ...
                              + grids.dt_trans(it) * steadystate.realgovdebt * (steadystate.r - fiscal_phi) ...
                              - grids.dt_trans(it) * shocks.transfershock(it)) ...
                              / (1 + grids.dt_trans(it) * (r(it) - fiscal_phi));
        end
    end

    realsurplus = steadystate.realsurplus + policy.fiscal_phi .* (realgovdebt - steadystate.realgovdebt);
end

%%
function out = splice_transition_structs(prefix,tail,start_idx)

    out = prefix;
    prefix_fields = fieldnames(prefix);
    tail_fields = fieldnames(tail);
    full_nT = numel(prefix.inflation);
    tail_nT = numel(tail.inflation);

    for ifield = 1:numel(tail_fields)
        name = tail_fields{ifield};
        tval = tail.(name);

        if iscell(tval) && isfield(prefix,name) && iscell(prefix.(name)) ...
                && numel(prefix.(name)) == full_nT && numel(tval) == tail_nT
            prefix_cell = reshape(prefix.(name),[],1);
            tail_cell = reshape(tval,[],1);
            out.(name) = [prefix_cell(1:start_idx-1); tail_cell];
        elseif isnumeric(tval) && isvector(tval) && isfield(prefix,name) && isnumeric(prefix.(name)) && isvector(prefix.(name))
            prefix_vec = prefix.(name)(:);
            tail_vec = tval(:);
            if numel(prefix.(name)) == full_nT && numel(tval) == tail_nT
                out.(name) = [prefix_vec(1:start_idx-1); tail_vec];
            elseif numel(prefix.(name)) == full_nT-1 && numel(tval) == tail_nT-1
                out.(name) = [prefix_vec(1:start_idx-1); tail_vec];
            else
                out.(name) = tval;
            end
        else
            out.(name) = tval;
        end
    end

    for ifield = 1:numel(prefix_fields)
        name = prefix_fields{ifield};
        if ~isfield(out,name)
            out.(name) = prefix.(name);
        end
    end

    out.res_assetmarket = out.Ea(2:end) - out.realgovdebt(2:end);
    out.res_labormarket = out.Eefflab - out.labor;
end

%%
function out = ternary(test,val_true,val_false)
    if test
        out = val_true;
    else
        out = val_false;
    end
end

%%
function [irf] = market_clearing_decomp(xdecomp,param,grids,steadystate)
     
    %extract some parameters and grids
    na = param.na;
    ny = param.ny;
    nT = param.nT;
    yygrid = grids.yygrid;
    
    %intialize arrays for HJB
    irf.V = cell(nT,1);
    irf.A = cell(nT,1);
    irf.con = cell(nT,1);
    irf.sav = cell(nT,1);
    irf.hour = cell(nT,1);
    
    % solve HJB backward from steady state

    for it = nT:-1:1
        
        passedvars.r = xdecomp.r(it);
        passedvars.realwage = xdecomp.realwage(it);
        passedvars.delta = grids.dt_trans(it);
        passedvars.divgrid = xdecomp.divgrid{it};
        passedvars.transfergrid = xdecomp.transfergrid{it};
        passedvars.labdisutil = steadystate.labdisutil;
        passedvars.rho = xdecomp.rho(it);
    
    
        if it==nT
            Vnext = steadystate.V;
        elseif it<nT
            Vnext = irf.V{it+1};
        end

        [irf.V{it},irf.A{it},irf.con{it},irf.sav{it},irf.hour{it}] = UpdateHJB(Vnext,param,grids,passedvars);
    
    end
    
    %initialize arrays for KFE
    irf.gmat = cell(nT,1);
    irf.gvecadj = cell(nT,1);

    irf.Ea = zeros(nT,1);
    irf.Eh = zeros(nT,1);
    irf.Ec = zeros(nT,1);
    irf.Eefflab = zeros(nT,1);

    %initialize at steady state;
    irf.gmat{1} = steadystate.gmat;
    irf.gvecadj{1} = reshape(grids.aydelta .* irf.gmat{1},na*ny,1);

    % solve KFE forward
    for it = 1:nT - 1
        irf.gvecadj{it+1} = (speye(na*ny) - irf.A{it}' * grids.dt_trans(it)) \ irf.gvecadj{it};
        irf.gmat{it+1}    = reshape(irf.gvecadj{it+1},na,ny) ./ grids.aydelta;
    end
 
    %asset supply
    for it = 1:nT
        irf.gamarg{it} = sum(irf.gmat{it},2);
        irf.Ea(it) = sum(irf.gamarg{it} .* grids.agrid .* grids.adelta);
        irf.Eh(it) = sum(irf.hour{it}(:).*irf.gmat{it}(:).*grids.aydelta(:));
        irf.Ec(it) = sum(irf.con{it}(:) .* irf.gmat{it}(:) .* grids.aydelta(:));
        irf.Eefflab(it) = sum(irf.hour{it}(:).* yygrid(:).*irf.gmat{it}(:).* grids.aydelta(:));
    end


end

%%
function irf = ra_market_clearing_decomp(xdecomp,param,grids,steadystate)
     
    %extract some parameters and grids
    nT = param.nT;
    
    %decomp variables
    irf.r = xdecomp.r;
    irf.dividend  = xdecomp.dividend;
    irf.realwage = xdecomp.realwage; 
    irf.transfer = xdecomp.transfer;
    irf.rho = xdecomp.rho;

    finalc = fsolve(@(x)fnradecomp(x,param,grids,steadystate,irf),steadystate.Ec);
    [~,irf] = fnradecomp(finalc,param,grids,steadystate,irf);
       
end

%%
function [Vnew,A,con,sav,hour] = UpdateHJB(V,param,grids,passedvars)

    %extract some variables
    na = param.na;
    ny = param.ny;
    agrid = grids.agrid;
    ygrid = grids.ygrid;
    yygrid = grids.yygrid;
    daagridf = grids.daagridf;
    daagridb = grids.daagridb;
    aagrid = grids.aagrid;
    yygrid = grids.yygrid;
    yytrans = grids.yytrans;
    r = passedvars.r;
    realwage = passedvars.realwage;
    delta = passedvars.delta;
    divgrid = passedvars.divgrid;
    transfergrid = passedvars.transfergrid;
    labdisutil = passedvars.labdisutil;
    rho = passedvars.rho;

    % initialize arrays as AD objects
    dVf = zeros(na,ny) + 0.*passedvars.r;
    dVb = zeros(na,ny) + 0.*passedvars.r;
    
    % forward difference         
    dVf(1:na-1,:) = (V(2:na,:)-V(1:na-1,:))./daagridf(1:na-1,:);
    dVf(na,:) = param.u1(r.*aagrid(na,:) + realwage.*yygrid(na,:) + transfergrid(na,:) + divgrid(na,:)); %state constraint, decisions updated below with endog lab supply

    % backward difference
    dVb(2:na,:) = (V(2:na,:)-V(1:na-1,:))./daagridb(2:na,:);
    dVb(1,:) = param.u1(r.*aagrid(1,:) + realwage.*yygrid(1,:) + transfergrid(1,:) + divgrid(1,:)); %state constraint, decisions updated below with endog lab supply

    %hours and consumption
    conf = param.u1inv(max(dVf,param.mindV));
    conb = param.u1inv(max(dVb,param.mindV));

    hourf = param.v1inv(realwage .*yygrid .* max(dVf,param.mindV),labdisutil);        
    hourf = min(hourf, param.maxhours);   
    hourb = param.v1inv(realwage .*yygrid .* max(dVb,param.mindV),labdisutil);        
    hourb = min(hourb, param.maxhours);        

    %forward difference, top point: solve non-linear equation by iteration
    for ih = 1:param.niter_hours
        conf(na,:)  = r.*aagrid(na,:) + hourf(na,:) .*realwage.* yygrid(na,:) + transfergrid(na,:) + divgrid(na,:);
        hourf(na,:) = param.dampen_hours.*hourf(na,:) + (1-param.dampen_hours) .* param.v1inv(param.u1(conf(na,:)) .* realwage.*yygrid(na,:),labdisutil) ;
        hourf(na,:) = min(hourf(na,:), param.maxhours);            
    end
    conf(na,:)  = max(r.*aagrid(na,:),0) + hourf(na,:) .*realwage.* yygrid(na,:) + transfergrid(na,:) + divgrid(na,:);
    dVf(na,:) = param.u1(conf(na,:));

    %backward difference, bottom point: solve non-linear equation by iteration
    for ih = 1:param.niter_hours
        conb(1,:)  = r.*aagrid(1,:) + hourb(1,:) .*realwage.* yygrid(1,:) + transfergrid(1,:) + divgrid(1,:);
        hourb(1,:) = param.dampen_hours.*hourb(1,:) + (1-param.dampen_hours) .*param.v1inv(param.u1(conb(1,:)) .* realwage.*yygrid(1,:) ,labdisutil) ;        
        hourb(1,:) = min(hourb(1,:), param.maxhours);
        
    end
    conb(1,:)  = r.*aagrid(1,:) + hourb(1,:) .*realwage.* yygrid(1,:) + transfergrid(1,:) + divgrid(1,:);
    dVb(1,:) = param.u1(conb(1,:));
    
    %zero points
    hour0 = hourb; % initialize
    for ih = 1:param.niter_hours
        con0  = max(r.*aagrid + hour0 .*realwage.* yygrid + transfergrid + divgrid,param.mindV);
        hour0 = param.dampen_hours.*hour0 + (1-param.dampen_hours) .*param.v1inv(param.u1(con0) .* realwage.*yygrid ,labdisutil) ;        
        hour0 = min(hour0, param.maxhours);    
    end
    con0  = max(r.*aagrid + hour0 .*realwage.* yygrid  + transfergrid + divgrid,param.mindV);
    dV0 = param.u1(con0);

    % nominal asset drift
    savf = r.*aagrid + (hourf.*realwage.*yygrid + transfergrid + divgrid - conf);
    savb = r.*aagrid + (hourb.*realwage.*yygrid + transfergrid + divgrid - conb);

    %hamiltonian
    Hf = param.u(conf) - param.v(hourf,labdisutil) + dVf.*savf;
    Hb = param.u(conb) - param.v(hourb,labdisutil) + dVb.*savb;
    H0 = param.u(con0) - param.v(hour0,labdisutil);    
    
    % choice of forward or backward differences based on sign of drift    
    Ineither = (1-(savf>0)) .* (1-(savb<0));
    Iunique = (savb<0).*(1-(savf>0)) + (1-(savb<0)).*(savf>0);
    Iboth = (savb<0).*(savf>0);
    Ib = Iunique.*(savb<0).*(Hb>H0) + Iboth.*(Hb>Hf).*(Hb>H0);
    If = Iunique.*(savf>0).*(Hf>H0) + Iboth.*(Hf>Hb).*(Hf>H0);
    I0 = 1-Ib-If;
    
    
    %consumption, savings and utility
    con  = conf.*If + conb.*Ib + con0.*I0;
    hour = hourf.*If + hourb.*Ib + hour0.*I0;    
    sav  = savf.*If + savb.*Ib;    
    util = param.u(con) - param.v(hour,labdisutil);
    utilvec = reshape(util,na*ny,1);
    
    %construct A matrix: tri-diagonal elements
    Alowdiag = -Ib.*savb./daagridb;
    Adiag = -If.*savf./daagridf + Ib.*savb./daagridb;
    Aupdiag = If.*savf./daagridf;
    
    
    %use spdiags to create A matrix for each income value
    Amat = cell(ny,1);
    for iy = 1:ny
        Amat{iy} = spdiags(Adiag(:,iy),0,na,na) + ...
                    spdiags(Alowdiag(2:na,iy),-1,na,na) + ...
                    spdiags([0;Aupdiag(1:na-1,iy)],1,na,na);
    end
    
    %combine to create large sparse A matrix
    if ny > 1
        A  = [Amat{1} sparse(na,na*(ny-1))];
        if ny>2    
            for iy = 2:ny-1
                A = [A; sparse(na,na*(iy-1)) Amat{iy} sparse(na,na*(ny-iy))];
            end
        end    
        A = [A; sparse(na,na*(ny-1)) Amat{ny}];
    elseif ny==1
        A = Amat{1};
    end
    
    % add Poisson income switches;
    A = A + yytrans;
    
    B = (rho + 1./delta)*speye(na*ny) - A;
    
    % solve linear system
    Vvec = reshape(V,na*ny,1);
    Vvecnew = B \ (utilvec + Vvec./delta);
    Vnew    = reshape(Vvecnew,na,ny);


end

%%
function [res,irf] = fnradecomp(lfinalc,param,grids,steadystate,irf)

    %extract some parameters and grids
    nT = param.nT;
    
    %steady state implied debt for that final consumption;
    lu1 = param.u1(lfinalc);        
    lfinalh = param.v1inv( lu1 .* steadystate.realwage, steadystate.labdisutil);
    lfinala = (lfinalc - lfinalh*steadystate.realwage - steadystate.transfer - steadystate.dividend)/steadystate.r;

    %solve Euler equation backward for u'(c)
    u1c = zeros(nT,1);
    irf.Ec = zeros(nT,1);

    u1c(nT) = param.u1(lfinalc);
    irf.Ec(nT) = param.u1inv(u1c(nT));
    for it = nT-1:-1:1
        u1c(it) = u1c(it+1) ./ (1 + (irf.rho(it) - irf.r(it)).*grids.dt_trans(it));
        irf.Ec(it) = param.u1inv(u1c(it));
    end

    %solve for hours
    irf.Eh = zeros(nT,1);
    for it = 1:nT
        lu1 = param.u1(irf.Ec(it));
        irf.Eh(it) = param.v1inv( lu1 .* irf.realwage(it) , steadystate.labdisutil);    
    end    
    irf.Eefflab = irf.Eh;

    %solve budget constraint forward for assets
    irf.Ea = zeros(nT,1);
    irf.Ea(1) = steadystate.Ea;

    %budget constraint
    irf.lflow = zeros(nT-1,1); %to hold flow in budget constraint
    for it = 1:nT-1
        irf.lflow(it) = irf.Eh(it).*irf.realwage(it) + irf.transfer(it) + irf.dividend(it) - irf.Ec(it);
        irf.Ea(it+1) = (irf.Ea(it) + grids.dt_trans(it).* irf.lflow(it)) ./ (1 - grids.dt_trans(it).*irf.r(it)); 
    end

    %residual
    res = irf.Ea(nT) - lfinala;
end    
%%
function PlotResultsSteadyState(fignum,param,grids,equm)
    % extract some objects
    agrid = grids.agrid;    
    ny = param.ny;
    na = param.na;
    amax = param.amax;
    borrow_lim = param.borrow_lim;
    con = equm.con;
    sav = equm.sav;
    gamarg = equm.gamarg;
    
    figure(fignum);
    
    % consumption policy function
    subplot(2,4,1);
    plot(agrid,con(:,1),'b-',agrid,con(:,ny),'r-','LineWidth',1);
    grid;
    xlim([borrow_lim amax]);
    title('Consumption Policy Function');
    legend('Lowest income state','Highest income state');

    % savings policy function
    subplot(2,4,2);
    plot(agrid,sav(:,1),'b-',agrid,sav(:,ny),'r-','LineWidth',1);
    hold on;
    plot(agrid,zeros(na,1),'k','LineWidth',0.5);
    hold off;
    grid;
    xlim([borrow_lim amax]);
    title('Savings Policy Function');
    
    % consumption policy function: zoomed in
    subplot(2,4,3);
    plot(agrid,con(:,1),'b-o',agrid,con(:,ny),'r-o','LineWidth',2);
    grid;
    xlim(borrow_lim + [0 1]);
    title('Consumption: Zoomed');
    
     % savings policy function: zoomed in
    subplot(2,4,4);
    plot(agrid,sav(:,1),'b-o',agrid,sav(:,ny),'r-o','LineWidth',2);
    hold on;
    plot(agrid,zeros(na,1),'k','LineWidth',0.5);
    hold off;
    grid;
    xlim(borrow_lim + [0 1]);
    title('Savings: Zoomed');
    
    %income distribution
    subplot(2,4,5);
    bar(grids.ygrid,grids.ydist);
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor',[0 0.5 0.5],'EdgeColor','blue','LineStyle','-');
    ylabel('')
    title('Income distribution');
    
    
    %asset distribution: CDF
    abin        = (borrow_lim:0.01:amax)'; %multiples of 1% of annual income
    nabin       = length(abin);
    
    interpacdf  = griddedInterpolant(agrid,cumsum(grids.adelta.*gamarg),'pchip');
    amass       = zeros(size(abin));
    amass(1)    = grids.adelta(1)*gamarg(1);
    amass(2:nabin) = interpacdf(abin(2:nabin)) - interpacdf(abin(1:nabin-1));
    
    % asset distribution: inverse CDF
    acdf = cumsum(grids.adelta.*gamarg);
    iacdf1 = find(acdf>0.9999999,1);
    if isempty(iacdf1)
        iacdf1 = na;
    end
    interpainvcdf  = griddedInterpolant(acdf(1:iacdf1),agrid(1:iacdf1),'linear');
    
    %plot asset distribution
    subplot(2,4,6:8);
    bar(abin(2:nabin),amass(2:nabin),'histc');
    sh  = findall(gcf,'marker','*'); delete(sh);
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor',[.7 .7 .7],'EdgeColor',[0.7 0.7 0.7],'LineStyle','-','LineWidth',0.01);
    
    xlim([param.borrow_lim agrid(min(iacdf1+1,na)) ]);
    
    hold on;
    bar(abin(1),amass(1),0.1,'FaceColor','black','EdgeColor','black','LineWidth',0.001);
    hold off;
    ylabel('')
    grid on;
    title('Asset distribution');
 
    
    disp(['Equilibrium interest rate: ' num2str(equm.r*100) '% (quarterly)']);
    disp(['Discount rate: ' num2str(equm.rho*100) '%  (quarterly)']);
    disp(['Mean assets: ' num2str(sum(gamarg.*grids.agrid.*grids.adelta))]);
    disp(['Fraction borrowing constrained: ' num2str(gamarg(1).*grids.adelta(1) * 100) '%']);
    disp(['10th Percentile: ' num2str(interpainvcdf(0.1))]);
    disp(['50th Percentile: ' num2str(interpainvcdf(0.5))]);
    disp(['90th Percentile: ' num2str(interpainvcdf(0.9))]);
    disp(['99th Percentile: ' num2str(interpainvcdf(0.99))]);

end

%%
function PlotResultsIRF(fignum,param,grids,steadystate,irf)
    %plots raw IRFs, i.e. no time aggregation

    % extract some objects
    Tplotmax = param.Tplotmax; %max plot in quarters
    Tgrid = grids.Tgrid;    
    nT = param.nT;
    
    % transition
    figure(fignum);    
    
    % tfp 
    subplot(2,4,1);
    hold on;
    plot(Tgrid,irf.tfp,'b-','LineWidth',1);
    plot(Tgrid,steadystate.tfp.*ones(nT,1),'k--','LineWidth',1);
    hold off;
    grid on;
    xlim([0 Tplotmax]);
    title('TFP');
    
    
    % assets
    subplot(2,4,2);
    hold on;
    plot(Tgrid,irf.realgovdebt,'b-','LineWidth',1);
    plot(Tgrid,irf.Ea,'r--','LineWidth',1);
    plot(Tgrid,steadystate.Ea.*ones(nT,1),'k--','LineWidth',1);
    hold off;
    grid on;
    xlim([0 Tplotmax]);
    legend('Govt Debt','Household Assets','Location','best')
    title('Assets');


    % output
    subplot(2,4,3);
    hold on;
    plot(Tgrid,irf.realoutput,'b-','LineWidth',1);
    plot(Tgrid,steadystate.realoutput.*ones(nT,1),'k--','LineWidth',1);
    hold off;
    grid on;
    xlim([0 Tplotmax]);
    title('Real Output');
    
    
    % consumption
    subplot(2,4,4);
    hold on;
    plot(Tgrid,irf.Ec,'b-','LineWidth',1);
    plot(Tgrid,steadystate.Ec.*ones(nT,1),'k--','LineWidth',1);
    hold off;
    grid on;
    xlim([0 Tplotmax]);
    title('Consumption');

    % interest rates and inflation
    subplot(2,4,5);
    hold on;
    plot(Tgrid,irf.r.*400,'b-','LineWidth',1);
    plot(Tgrid,irf.rnom.*400,'r-','LineWidth',1);
    plot(Tgrid,irf.inflation.*400,'k-','LineWidth',1);
    plot(Tgrid,steadystate.r.*ones(nT,1).*400,'k--','LineWidth',1);
    hold off;
    grid on;
    xlim([0 Tplotmax]);
    legend('Real Rate','Nominal Rate','Inflation','Location','best')
    title('Interest Rates (% p.a.)');

    % wage rate
    subplot(2,4,6);
    hold on;
    plot(Tgrid,irf.realwage,'b-','LineWidth',1);
    plot(Tgrid,steadystate.realwage.*ones(nT,1),'k--','LineWidth',1);
    hold off;
    grid;
    xlim([0 Tplotmax]);
    title('Real Wage');

    % gov surpluses
    subplot(2,4,7);
    hold on;
    plot(Tgrid,irf.realsurplus,'b-','LineWidth',1);
    plot(Tgrid,steadystate.realsurplus.*ones(nT,1),'k--','LineWidth',1);
    hold off;
    grid;
    xlim([0 Tplotmax]);
    title('Real Surplus');


end

%%
function PlotResultsCompareRA(fignum,grids,param,ra_param,steadystate,ra_steadystate,irf, ra_irf)
        %plots raw IRFs, i.e. no time aggregation

    % extract some objects
    Tplotmax = param.Tplotmax; %max plot in quarters
    Tgrid = grids.Tgrid;    
    nT = param.nT;
    
    % transition
    figure(fignum);    
    
    % output
    subplot(2,4,1);
    hold on;
    plot(Tgrid,irf.realoutput./steadystate.realoutput,'b-','LineWidth',2);
    plot(Tgrid,ra_irf.realoutput./ra_steadystate.realoutput,'r--','LineWidth',2);
    plot(Tgrid,ones(nT,1),'k--','LineWidth',1);
    hold off;
    grid on;
    xlim([0 Tplotmax]);
    legend('Het Agent','Rep Agent','Location','best')
    title('Real Output');
    
    % consumption
    subplot(2,4,2);
    hold on;
    plot(Tgrid,irf.Ec./steadystate.Ec,'b-','LineWidth',2);
    plot(Tgrid,ra_irf.Ec./ra_steadystate.Ec,'r--','LineWidth',2);
    plot(Tgrid,ones(nT,1),'k--','LineWidth',1);
    hold off;
    grid on;
    xlim([0 Tplotmax]);
    legend('Het Agent','Rep Agent','Location','best')
    title('Consumption');       
    
    % assets
    subplot(2,4,3);
    hold on;
    plot(Tgrid,irf.realgovdebt./steadystate.realoutput,'b-','LineWidth',2);
    plot(Tgrid,ra_irf.realgovdebt./ra_steadystate.realoutput,'r--','LineWidth',2);
    plot(Tgrid,steadystate.realgovdebt./steadystate.realoutput.*ones(nT,1),'k--','LineWidth',1);
    hold off;
    grid on;
    xlim([0 Tplotmax]);
    legend('Het Agent','Rep Agent','Location','best')
    title('Assets');


    % gov surpluses (without transfer shocks these are the same thing)
    subplot(2,4,4);
    hold on;
    % plot(Tgrid,irf.realsurplus./steadystate.realoutput,'b-','LineWidth',2);  
    % plot(Tgrid,ra_irf.realsurplus./ra_steadystate.realoutput,'r--','LineWidth',2);
    plot(Tgrid,-irf.transfer./steadystate.realoutput,'b-','LineWidth',2);
    plot(Tgrid,-ra_irf.transfer./ra_steadystate.realoutput,'r--','LineWidth',2);
    plot(Tgrid,steadystate.realsurplus./steadystate.realoutput.*ones(nT,1),'k--','LineWidth',1);
    hold off;
    grid;
    xlim([0 Tplotmax]);
    legend('Het Agent','Rep Agent','Location','best')
    title('Real Surplus');

    % inflation
    subplot(2,4,5);
    hold on;
    plot(Tgrid,irf.inflation.*400,'b-','LineWidth',2);
    plot(Tgrid,ra_irf.inflation.*400,'r:','LineWidth',2);
    plot(Tgrid,steadystate.inflation.*ones(nT,1).*400,'k--','LineWidth',1);
    hold off;
    grid on;
    xlim([0 Tplotmax]);
    legend('Het Agent','Rep Agent','Location','best')
    title('Inflation (% p.a.)');

    % real rate
    subplot(2,4,6);
    hold on;
    plot(Tgrid,irf.r.*400,'b-','LineWidth',2);
    plot(Tgrid,ra_irf.r.*400,'r:','LineWidth',2);
    plot(Tgrid,steadystate.r.*ones(nT,1).*400,'k--','LineWidth',1);
    hold off;
    grid on;
    xlim([0 Tplotmax]);
    legend('Het Agent','Rep Agent','Location','best')
    title('Real Rate (% p.a.)');


    % nominal rate
    subplot(2,4,7);
    hold on;
    plot(Tgrid,irf.rnom.*400,'b-','LineWidth',2);
    plot(Tgrid,ra_irf.rnom.*400,'r:','LineWidth',2);
    plot(Tgrid,steadystate.rnom.*ones(nT,1).*400,'k--','LineWidth',1);
    hold off;
    grid on;
    xlim([0 Tplotmax]);
    legend('Het Agent','Rep Agent','Location','best')
    title('Nominal Rate (% p.a.)');

    % real wage
    subplot(2,4,8);
    hold on;
    plot(Tgrid,irf.realwage./steadystate.realwage,'b-','LineWidth',2);
    plot(Tgrid,ra_irf.realwage./ra_steadystate.realwage,'r:','LineWidth',2);
    plot(Tgrid,ones(nT,1),'k--','LineWidth',1);
    hold off;
    grid on;
    xlim([0 Tplotmax]);
    legend('Het Agent','Rep Agent','Location','best')
    title('Real Wage');       
        
end

%%
function [f,x,p] = discrete_normal(n,mu,sigma,width)
% creates equally spaced approximation to normal distribution
% n is number of points
% mu is mean
% sigma is standard deviation
% width is the multiple of stand deviation for the width of the grid
% f is the error in the approximation
% x gives the location of the points
% p is probabilities


x = linspace(mu-width*sigma,mu+width*sigma,n)';

if n==2
    p = 0.5.*ones(n,1);
elseif n>2    
    p  = zeros(n,1);
    p(1) = normcdf(x(1) + 0.5*(x(2)-x(1)),mu,sigma);
    for i = 2:n-1
        p(i) = normcdf(x(i) + 0.5*(x(i+1)-x(i)),mu,sigma) - normcdf(x(i) - 0.5*(x(i)-x(i-1)),mu,sigma);
    end
    p(n) = 1 - sum(p(1:n-1));
end

Ex = x'*p;
SDx = sqrt((x'.^2)*p - Ex.^2);

f = SDx-sigma;
end

%% functions
function [val,jac] = fAD(x,f)
    xAD = myAD(x);
    resAD = f(xAD);
    val = getvalues(resAD);
    jac = getderivs(resAD);
end
