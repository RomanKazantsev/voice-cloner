% List all files for training and time ranges for parallel data
names{1} = 'arctic_a0001.wav';

ts = [0.65, 1.05, 1.12, 1.20, 1.30, 1.65, 1.70, 2.1, 2.8, 3.2, 2.35, 2.65, 2.65, 2.75];
tt = [0.20, 0.70, 0.80, 0.88, 0.90, 1.25, 1.25, 1.7, 2.1, 2.6, 1.80, 2.00, 2.04, 2.13];

num_files = length(names);

melfcc_s = [];
melfcc_t = [];
f0_t = [];

for i = 1:num_files
  cd source
  [s, fs] = audioread(names{i});
  cd ..
  cd target
  [t, fs] = audioread(names{i});
  cd ..
  
  [tmp_melfcc_s, tmp_melfcc_t] = get_parallel_data(s, t, fs, ts, tt);
  
  melfcc_s = [melfcc_s tmp_melfcc_s];
  melfcc_t = [melfcc_t tmp_melfcc_t];
  
  
  tmp_f0_t = Harvest(t, fs);
  f0_t = [f0_t tmp_f0_t.f0];
end

f0_t
f0_t = f0_t(f0_t~=0);
mean_logf0 = mean(log(f0_t));
var_logf0 = var(log(f0_t));

dlmwrite('s_melfcc_train', melfcc_s', ' ');
dlmwrite('t_melfcc_train', melfcc_t', ' ');
dlmwrite('t_mean_logf0', mean_logf0', ' ');
dlmwrite('t_var_logf0', var_logf0', ' ');
