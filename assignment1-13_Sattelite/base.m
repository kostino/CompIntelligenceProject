Gp_tf = zpk([],[-1 -9],10); % Gp(s)

Gc_tf = zpk(-1.1 , 0 , 1); % mideniko konta sto -1 

controlSystemDesigner(Gp_tf,Gc_tf);