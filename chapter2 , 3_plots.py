"""##fig 2.1"""


import numpy as np
import matplotlib.pyplot as plt

# رویدادها (t1..t7). t5 و t6 کمی نزدیک هم قرار گرفته‌اند
event_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.3, 7.0])

# مقادیر N(t) **بعد** از هر جهش
post_values = np.arange(1, len(event_times) + 1)

# برای رسم پله‌ای فقط با خطوط افقی
times = np.concatenate(([0.0], event_times, [event_times[-1] + 0.6]))
values = np.concatenate(([0], post_values))

plt.figure(figsize=(10, 3.6))

# رسم خطوط افقی برای هر پله
for i in range(len(values) - 1):
    plt.hlines(y=values[i], xmin=times[i], xmax=times[i+1], linewidth=3, color="#5ab4ac")

# نقاط توپر
plt.plot(event_times, post_values, 'o', markersize=10, color="#5ab4ac")

# نقاط توخالی
pre_values = post_values - 1
plt.plot(event_times, pre_values, 'o', markersize=10,
         markerfacecolor='white', markeredgecolor='#5ab4ac')

# نقطه توخالی در t=0 برای N(0)=0
plt.plot(0, 0, 'o', markersize=10, markerfacecolor='white', markeredgecolor='#5ab4ac')

# برچسب‌ها و تنظیمات ظاهری
plt.xlabel('t')
plt.ylabel('N(t)')
plt.xticks(list(event_times), [f'$t_{{{i}}}$' for i in range(1, len(event_times)+1)])
plt.xlim(-0.2, times[-1])
plt.ylim(-0.5, post_values[-1] + 1.0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

"""##fig2.2"""

import numpy as np
import matplotlib.pyplot as plt

# زمان‌های رویدادها (t1..t7)
event_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.3, 8.5])

# پارامترها
base_intensity = 0.5
jump_height = 1.0
decay_rate = 1.0

# بازه زمانی
t_min = 0.0
t_max = event_times[-1] + 1.0
t = np.linspace(t_min, t_max, 2000)

# تابع شدت
intensity = np.full_like(t, base_intensity)
for te in event_times:
    mask = t >= te
    intensity[mask] += jump_height * np.exp(-decay_rate * (t[mask] - te))

# pre و post
pre_values, post_values = [], []
for i, te in enumerate(event_times):
    pre = base_intensity
    for prev in event_times[:i]:
        pre += jump_height * np.exp(-decay_rate * (te - prev))
    post = pre + jump_height
    pre_values.append(pre)
    post_values.append(post)

# ---- رسم ----
plt.figure(figsize=(9, 3.5))
color = "#5ab4ac"
# خط پایه
plt.hlines(base_intensity, t_min, event_times[0], color=color, linewidth=4)

# منحنی‌های نمایی بین رویدادها
intervals = np.concatenate((event_times, [t_max]))
for i, start in enumerate(event_times):
    end = intervals[i + 1]
    mask = (t >= start) & (t <= end)
    plt.plot(t[mask], intensity[mask], color=color, linewidth=4)

# نقاط pre (توخالی) و post (توپر)
for te, pre, post in zip(event_times, pre_values, post_values):
    plt.plot(te, pre, 'o', markersize=10, markerfacecolor='white',
             markeredgecolor=color, markeredgewidth=1.5)
    plt.plot(te, post, 'o', markersize=10, color=color)

# برچسب محورها
plt.xlabel(r"$t$", fontsize=14)
plt.ylabel(r"$\lambda^*(t)$", fontsize=14)

# برچسب زمان‌ها
plt.xticks(event_times, [f"$t_{{{i+1}}}$" for i in range(len(event_times))], fontsize=12)
plt.yticks(fontsize=12)

# حذف خطوط اضافی
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# متن λ کنار خط پایه
plt.text(t_min - 0.3, base_intensity, r"$\lambda$", fontsize=14, ha="right", va="center")

plt.xlim(t_min - 0.5, t_max + 0.5)
plt.ylim(0, max(post_values) + 1.0)
plt.tight_layout()
plt.show()

"""##fig3.2"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- پارامترها ----------
T = 100.0
mu = 0.4
alpha = 0.9
beta = 1.5
np.random.seed(7)

# ---------- شبیه‌سازی فرآیند هاوکز (Ogata thinning) ----------
events = []
t = 0.0
lam_curr = mu
lam_bar = lam_curr

while t < T:
    t += -np.log(np.random.rand()) / max(lam_bar, 1e-12)
    if t > T:
        break
    if events:
        diffs = t - np.array(events)
        lam_t = mu + np.sum(alpha * np.exp(-beta * diffs[diffs >= 0]))
    else:
        lam_t = mu
    if np.random.rand() <= lam_t / max(lam_bar, 1e-12):
        events.append(t)
        lam_bar = lam_t + alpha
        lam_curr = lam_t + alpha
    else:
        lam_bar = lam_t
        lam_curr = lam_t

events = np.array(events)

# ---------- محاسبه کمیت‌ها ----------
grid = np.linspace(0, T, 4001)
N_t = np.searchsorted(events, grid, side="right")
if len(events):
    lam_star = mu + (alpha * np.exp(-beta * (grid[:, None] - events[None, :]))
                     * (grid[:, None] >= events[None, :])).sum(axis=1)
else:
    lam_star = np.full_like(grid, mu, dtype=float)

n = alpha / beta
E_lambda = mu / (1.0 - n)
E_N = E_lambda * grid

# ---------- استایل ----------
color_main = "#5ab4ac"   # سبز-آبی
color_mean = "#f46d6d"   # قرمز ملایم

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.default": "it",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                               gridspec_kw=dict(hspace=0.25))

# (a) Count
ax1.step(grid, N_t, where="post", linewidth=2.5, color=color_main, label=r"$N(t)$")
ax1.plot(grid, E_N, "--", linewidth=2.5, color=color_mean, label=r"$\mathbb{E}[N(t)]$")
ax1.set_ylabel("Count")
ax1.legend(loc="upper left", frameon=False, fontsize=12, handlelength=2.5)
ax1.set_xlim(0, T)
ax1.set_ylim(0, 100)

# (b) Intensity
ax2.plot(grid, lam_star, linewidth=2.5, color=color_main, label=r"$\lambda^*(t)$")
ax2.hlines(E_lambda, 0, T, linestyles="--", linewidth=2.5, color=color_mean,
           label=r"$\mathbb{E}[\lambda^*(t)]$")
ax2.set_ylabel("Intensity")
ax2.set_xlabel(r"$t$")
ax2.set_ylim(0, max(3.5, lam_star.max()*1.05))
ax2.legend(loc="upper left", frameon=False, fontsize=12, handlelength=2.5)

plt.tight_layout()
plt.show()

"""##fig4.2"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

baseline_y = 122

def H(x_pix, y_pix):
    return baseline_y - y_pix

nodes = {
    "I1": dict(x=149, h=H(149, 43),  kind="imm"),
    "C1": dict(x=205, h=H(205, 80),  kind="offs"),
    "C2": dict(x=250, h=H(250, 10),  kind="offs"),
    "C3": dict(x=306, h=H(306, 45),  kind="offs"),
    "C4": dict(x=323, h=H(323, 10),  kind="offs"),
    "I2": dict(x=441, h=H(441, 43),  kind="imm"),
    "C5": dict(x=606, h=H(606,100),  kind="offs"),
    "C6": dict(x=652, h=H(652, 49),  kind="offs"),
    "I3": dict(x=688, h=H(688, 10),  kind="imm"),
    "C7": dict(x=743, h=H(743, 45),  kind="offs"),
}

edges = [
    ("I1", "C2"),
    ("I1", "C1"),
    ("C2", "C3"),
    ("C2", "C4"),
    ("I2", "C5"),
    ("C5", "C6"),
    ("I2", "C7"),
]

crosses_x = [149, 205, 249, 304, 322, 441, 606, 652, 688, 736, 787]

xmin, xmax = 90, 840
hmax = max(n["h"] for n in nodes.values())
ymin, ymax = -5, hmax + 15

fig, ax = plt.subplots(figsize=(9.5, 2.4), dpi=200)

# محور زمان
ax.hlines(0, xmin, xmax-22, linewidth=1.2, color="black")
ax.arrow(xmax-22, 0, 18, 0, head_width=6, head_length=10,
         linewidth=1.2, color="black", length_includes_head=True)
ax.text(xmax-2, -2.5, r"$t$", fontsize=12, ha="left", va="top")

# ×
def draw_cross(x, y=0, size=8, lw=1.2):
    d = size/2
    ax.plot([x-d, x+d], [y-d, y+d], color="#af7c74", lw=lw)
    ax.plot([x-d, x+d], [y+d, y-d], color="#af7c74", lw=lw)

for x in crosses_x:
    draw_cross(x, 0, size=9, lw=1.2)

# خط‌چین‌ها
for nd in nodes.values():
    ax.vlines(nd["x"], 0, nd["h"], linestyles=(0, (2, 4)),
              colors="#33CCFF", lw=0.9)

# گره‌ها
for name, nd in nodes.items():
    if nd["kind"] == "imm":
        ax.scatter([nd["x"]], [nd["h"]], s=75, marker="s",
                   color="#294D3C", zorder=3)  # مربع بزرگ‌تر
    else:
        ax.scatter([nd["x"]], [nd["h"]], s=55, marker="o",
                   color="#46B381", zorder=3)

# فلش‌ها
def arrow(p, q, shrink=6, lw=1.2):
    x0, y0 = nodes[p]["x"], nodes[p]["h"]
    x1, y1 = nodes[q]["x"], nodes[q]["h"]
    v = np.array([x1-x0, y1-y0], dtype=float)
    L = np.hypot(*v)
    if L > 1e-6:
        v /= L
        x0s = x0 + shrink*v[0]
        y0s = y0 + shrink*v[1]
        x1s = x1 - shrink*v[0]
        y1s = y1 - shrink*v[1]
    else:
        x0s, y0s, x1s, y1s = x0, y0, x1, y1

    patch = FancyArrowPatch(
        (x0s, y0s), (x1s, y1s),
        arrowstyle="-|>", mutation_scale=10,
        linewidth=1.3, color="#f46d6d", zorder=2
    )
    ax.add_patch(patch)

for u, v in edges:
    arrow(u, v, shrink=9, lw=1.3)

# تنظیمات نهایی
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xticks([]); ax.set_yticks([])
for spine in ["top", "left", "right", "bottom"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()

"""##fig2.3"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# ----------------------
# پارامترهای توزیع لگ‌نرمال
# ----------------------
mean_days = 4.7   # میانگین
sd_days = 2.9     # انحراف معیار

# تبدیل به پارامترهای توزیع نرمال معادل (m, s)
s = np.sqrt(np.log(1 + (sd_days**2 / mean_days**2)))
m = np.log(mean_days) - 0.5 * s**2

# تولید داده تصادفی برای رسم هیستوگرام
rng = np.random.default_rng(42)
samples = rng.lognormal(mean=m, sigma=s, size=150000)

# ----------------------
# رسم نمودار
# ----------------------
fig, ax = plt.subplots(figsize=(12, 5), dpi=220)

# هیستوگرام با میله‌های سفید و قاب مشکی
bin_width = 1.5
bins = np.arange(0, 30 + bin_width, bin_width)
ax.hist(samples, bins=bins, density=True,
        edgecolor="black", facecolor="white", linewidth=1.8)

# رسم تابع چگالی تئوری لگ‌نرمال (خط ماژنتا)
x = np.linspace(0, 30, 1200)
pdf = lognorm.pdf(x, s=s, scale=np.exp(m))
ax.plot(x, pdf, color="magenta", linewidth=3.0)

# ----------------------
# تنظیمات ظاهری
# ----------------------
for spine in ax.spines.values():
    spine.set_linewidth(2.2)
    spine.set_color("black")

# محور x: تیک‌ها هر ۱۰ واحد یکبار
ax.set_xlim(0, 30)
ax.set_xticks([0, 10, 20, 30])
ax.set_xticklabels(["0", "10", "20", "30"], fontsize=12)

# محور y: بدون تیک
ax.set_yticks([])

# برچسب‌ها
ax.set_ylabel(r"$\phi_d$", fontsize=18, rotation=0,
              labelpad=25, va="center")
ax.yaxis.set_label_coords(-0.04, 0.5)

ax.set_xlabel("d", fontsize=16, x=1.0, ha="right")

plt.tight_layout()
plt.show()
