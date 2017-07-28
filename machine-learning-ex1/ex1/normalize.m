function N = normalize(x)
m = length(x);
mean_value = sum(x)/m ;
N = (x-ones(m,1)*mean_value)/(max(x)-min(x));
end


