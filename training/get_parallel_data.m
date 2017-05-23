function [melfcc_s, melfcc_t] = get_parallel_data(s, t, fs, time_s, time_t)
%GET_PARALLEL_DATA returns parallel mfcc data of s1 and s2 audio signals
%with sampleing rate sr
assert(mod(length(time_s),2) == 0);
assert(length(time_s) == length(time_t));

source_duration = length(s) / fs;
target_duration = length(t) / fs;

melfcc_s = [];
melfcc_t = [];

i = 1;

while (i <= length(time_s))
    i
    s_start_ind = floor(length(s) * time_s(i) / source_duration);
    s_end_ind = floor(length(s) * time_s(i + 1) / source_duration);
    t_start_ind = floor(length(t) * time_t(i) / target_duration);
    t_end_ind = floor(length(t) * time_t(i + 1) / target_duration);

    s_start_ind = check_and_fix_ind(s_start_ind, 1, length(s));
    s_end_ind = check_and_fix_ind(s_end_ind, 1, length(s));
    t_start_ind = check_and_fix_ind(t_start_ind, 1, length(t));
    t_end_ind = check_and_fix_ind(t_end_ind, 1, length(t));
    
    ss =  s(s_start_ind:s_end_ind);
    tt = t(t_start_ind:t_end_ind);
    
    [scepstra] = melfcc(ss, fs, 'numcep', 13, 'lifter', 0, 'maxfreq', 8000);
    [tcepstra] = melfcc(tt, fs, 'numcep', 13, 'lifter', 0, 'maxfreq', 8000);
    
    [scepstra, tcepstra] = process_with_dtw(scepstra, tcepstra);
    
    melfcc_s = [melfcc_s scepstra(2:13,:)];
    melfcc_t = [melfcc_t tcepstra(2:13,:)];
    
    i = i + 2;
end

end

function [ind] = check_and_fix_ind(ind, r1, r2)
if(ind < r1)
   ind = r1; 
end
if(ind > r2)
   ind = r2; 
end
end

function [sscepstra, ttcepstra] = process_with_dtw(scepstra, tcepstra)
  % prediction male cepstra coeffs
  [dist, ix, iy] = dtw(scepstra, tcepstra);

  cur1 = 0;
  cur2 = 0;
  sscepstra = [];
  ttcepstra = [];
  for j = 1:size(ix)
    if cur1 >= ix(j)
        continue
    end
    if cur2 >= iy(j)
        continue
    end
    cur1 = ix(j);
    cur2 = iy(j);
    sscepstra = [sscepstra scepstra(:,cur1)];
    ttcepstra = [ttcepstra tcepstra(:,cur2)];
  end

end
