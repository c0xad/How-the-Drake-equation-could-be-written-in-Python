def drake_equation(R, fp, ne, fl, fi, fc, L):
    N = R * fp * ne * fl * fi * fc * L
    return N

R = 10 # average rate of star formation per year in our galaxy
fp = 0.1 # fraction of stars that have planets
ne = 2 # average number of planets that can potentially support life per star that has planets
fl = 0.01 # fraction of planets that actually develop life
fi = 0.001 # fraction of planets that develop intelligent life
fc = 0.01 # fraction of civilizations that develop a technology that releases detectable signs of their existence into space
L = 100 # length of time such civilizations release detectable signals into space

N = drake_equation(R, fp, ne, fl, fi, fc, L)
print(N)
