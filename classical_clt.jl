# classical central limit theorem 

using Distributions 
using Random
using Gnuplot

function clt(maxn) 
    μ = 10
    D = Normal(μ, 1)    
    savg = zeros(Float64, maxn)
    for n = 1:maxn 
        xbar = mean(rand(D, n))
        savg[n] = sqrt(n)*(xbar - μ)
    end 
    savg
end

function plothist(x)
    h = hist(x, bs=0.05)
    bs = fill(h.binsize, length(h.bins));
    println(h.counts)
    @gp h.bins h.counts./2 bs./2 h.counts./2 "w boxxy notit fs solid 0.4" 
end