# verify every ground truth for the new hard dataset
import math

print("=== A. optimization ===")
# A1 cylinder V=500 closed
r = (250 / math.pi) ** (1 / 3)
h = 500 / (math.pi * r ** 2)
S = 2 * math.pi * r ** 2 + 2 * math.pi * r * h
print(f"A1 cylinder: r={r:.3f} h={h:.3f} S={S:.2f} (h=2r? {h/r:.4f})")
# A2 fence 600m against river
x = 150; w = 600 - 2 * x
print(f"A2 fence: depth=150, width={w}, area={x*w}")
# A3 open box from 24x24, cut x
best = max(((24 - 2 * x) ** 2 * x, x) for x in [i / 1000 for i in range(1, 12000)])
print(f"A3 box: V*={best[0]:.1f} at x={best[1]:.3f} (analytic x=4, V={(24-8)**2*4})")
# A4 x+y=30 max x*y^2
best = max((x * (30 - x) ** 2, x) for x in [i / 1000 for i in range(1, 30000)])
print(f"A4: max={best[0]:.1f} at x={best[1]:.3f} (analytic x=10,y=20, 10*400=4000)")
# A5 sliding ladder 5m, bottom at 3m moving 1 m/s
y = math.sqrt(25 - 9); dydt = -3 * 1 / y
print(f"A5 ladder: y={y}, dy/dt={dydt} m/s")

print("\n=== B. probability ===")
# B1 expected flips HH=6, HT=4 (standard results)
# simulate to confirm
import random
random.seed(1)
def sim(pattern, n=200000):
    tot = 0
    for _ in range(n):
        s = ""
        while pattern not in s:
            s += random.choice("HT")
            if len(s) > 200: break
        tot += len(s)
    return tot / n
print(f"B1: HH={sim('HH'):.2f} (exact 6), HT={sim('HT'):.2f} (exact 4)")
# B2 gambler's ruin start 3 target 10 fair: p=3/10, E[duration]=3*(10-3)=21
print("B2: p(win)=3/10=0.3, expected duration=3*7=21")
# B3 coupon collector n=6
e = 6 * sum(1 / k for k in range(1, 7))
print(f"B3 coupon: {e:.2f} (=14.7)")
# B4 E[max of two dice]
em = sum(max(a, b) for a in range(1, 7) for b in range(1, 7)) / 36
print(f"B4 E[max]: {em:.4f} = 161/36 = {161/36:.4f}")
# B5 expected draws until first ace = 53/5
print(f"B5 first ace: 53/5 = {53/5}")

print("\n=== C. physics (g=9.8) ===")
# C1 incline 30 deg, mu_k=0.2, slide 5 m
a = 9.8 * (math.sin(math.radians(30)) - 0.2 * math.cos(math.radians(30)))
t = math.sqrt(2 * 5 / a)
print(f"C1: a={a:.3f} m/s^2, t={t:.3f} s")
# C2 cliff 20 m, v0=15 horizontal
tf = math.sqrt(2 * 20 / 9.8); rng = 15 * tf; vy = 9.8 * tf; vi = math.hypot(15, vy)
print(f"C2: t={tf:.3f} s, range={rng:.2f} m, impact speed={vi:.2f} m/s")
# C3 spring k=200 x=0.1 m=0.5
v = math.sqrt(200 * 0.01 / 0.5); hh = v ** 2 / (2 * 9.8)
print(f"C3: v={v} m/s, h={hh:.4f} m")
# C4 curve r=50 mu=0.6
vmax = math.sqrt(0.6 * 9.8 * 50)
print(f"C4: vmax={vmax:.2f} m/s = {vmax*3.6:.1f} km/h")
# C5 RC 10k 100uF -> tau=1s, V(2s), t to 6V
tau = 10000 * 100e-6
v2 = 12 * (1 - math.exp(-2 / tau)); t6 = tau * math.log(2)
print(f"C5: tau={tau} s, V(2s)={v2:.2f} V, t(6V)={t6:.3f} s")

print("\n=== D. finance ===")
# D1 loan 100k, 6%/yr monthly, 15 yr
r = 0.06 / 12; n = 180
pay = 100000 * r / (1 - (1 + r) ** -n)
print(f"D1: payment={pay:.2f}, total interest={pay*n-100000:.2f}")
# D2 NPV 10k cost, 4k x 3yr, 8%
npv = sum(4000 / 1.08 ** k for k in range(1, 4)) - 10000
print(f"D2: NPV={npv:.2f}")
# D3 doubling at 7%
td = math.log(2) / math.log(1.07)
print(f"D3: exact={td:.3f} yr, rule of 72={72/7:.3f} yr")
# D4 FV annuity 200/mo 10yr 6%
r = 0.005; n = 120
fv = 200 * ((1 + r) ** n - 1) / r
print(f"D4: FV={fv:.2f}, contributed={200*120}, interest={fv-24000:.2f}")
# D5 EAR
ear_m = 1.01 ** 12 - 1; ear_q = 1.03 ** 4 - 1
print(f"D5: monthly EAR={ear_m*100:.4f}%, quarterly EAR={ear_q*100:.4f}%, diff={(ear_m-ear_q)*100:.4f}pp")

print("\n=== E. discrete ===")
# E1 CRT x=3 mod5, 4 mod7, 1 mod9
x = next(x for x in range(1, 5*7*9+1) if x % 5 == 3 and x % 7 == 4 and x % 9 == 1)
print(f"E1 CRT: x={x} (mod {5*7*9})")
# E2 7^222 mod 11
print(f"E2: {pow(7,222,11)}")
# E3 phi(360)
def phi(n):
    res, p, nn = n, 2, n
    while p * p <= nn:
        if nn % p == 0:
            while nn % p == 0: nn //= p
            res -= res // p
        p += 1
    if nn > 1: res -= res // nn
    return res
print(f"E3: phi(360)={phi(360)}, not coprime={360-phi(360)}")
# E4 a(n)=3a(n-1)-2a(n-2), a0=2 a1=5 -> 3*2^n-1
a = [2, 5]
for i in range(2, 11): a.append(3 * a[-1] - 2 * a[-2])
print(f"E4: a(10)={a[10]}, closed form check 3*2^10-1={3*1024-1}")
# E5 base 7: 452_7 + 266_7
v = int("452", 7) + int("266", 7)
def to_base(n, b):
    s = ""
    while n: s = str(n % b) + s; n //= b
    return s
print(f"E5: 452_7+266_7 = {v} decimal = {to_base(v,7)}_7")
