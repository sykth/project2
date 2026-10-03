"""
세 가지 옥상 시나리오(일반 / 쿨루프 / 옥상 녹화)를 계산해서 data/results.npz 로 저장.
설정값(uhi_core.py)을 바꿨을 때만 다시 실행하면 돼.

    python precompute.py
"""
import time
import uhi_core as C

if __name__ == "__main__":
    t = time.time()
    print("계산 시작 (코어 수에 따라 1~3분)...")
    res = C.compute_all(C.DAYS, parallel=True)
    C.save_results(res)
    print(f"완료: {C.DATA_PATH}  ({time.time() - t:.0f}초)")
    for k in C.ROOF_KEYS:
        r = res[k]
        d = r["u_air"] - r["f_air"]
        print(f"  {k:9s} 열섬 최대 {d.max():.2f} °C | 옥상 최고 {r['rT'].mean(axis=1).max():.1f} °C")
