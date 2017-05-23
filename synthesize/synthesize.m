name  = 'source.wav';

[s, fs] = audioread(name);

f0_parameter = Harvest(s, fs);
s_f0 = f0_parameter.f0;
ap = D4C(s, fs, f0_parameter);

s_mean_logf0 = mean(log(s_f0(s_f0~=0)));
s_var_logf0 = var(log(s_f0(s_f0~=0)));
t_mean_logf0 = dlmread('t_mean_logf0', ' ');
t_var_logf0 = dlmread('t_var_logf0', ' ');

for i = 1:length(s_f0)
   if s_f0(i) ~= 0
       s_logf0 = log(s_f0(i));
       s_logf0 = (s_logf0 - s_mean_logf0) / s_var_logf0 * t_var_logf0 + t_mean_logf0;
       s_f0(i) = exp(s_logf0);
   end
end

t_f0_parameter = f0_parameter;
t_f0_parameter.f0 = s_f0;
ap.f0 = s_f0;

[scepstra] = melfcc(s, fs, 'numcep', 13, 'lifter', 0, 'maxfreq', 8000);
s_melfcc0 = scepstra(1:1,:);
t_melfcc = (dlmread('t_melfcc_predict', ' '))';
t_melfcc = [s_melfcc0; t_melfcc];

[t_invmelfcc] = invmelfcc(t_melfcc, fs, 'numcep', 13, 'lifter', 0, 'maxfreq', 8000);

t_spectrum_parameter = CheapTrick(t_invmelfcc, fs, t_f0_parameter);

z = Synthesis(ap, t_spectrum_parameter);

soundsc(z, fs);

audiowrite('target.wav', z, fs);

