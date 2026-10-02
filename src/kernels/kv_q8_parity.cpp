// src/kernels/kv_q8_parity.cpp - plan v0.3 P7: INT8 KV append/gather against a host reference (GPU, no model).
//
// Appends random K/V cells at scattered positions through a non-identity page table into both the INT8 pools
// (kv_append_q8_step) and the FP16 pools (kv_append_step), gathers random cell sets from both, and checks:
//   1. INT8 codes and FP16 scales are BITWISE equal to the host reference (same f16_from_f32 rounding);
//   2. the INT8 gather is BITWISE equal to the host dequantization of those codes;
//   3. against the FP16 path, the INT8 values differ by at most half a quantization step of their group plus the
//      fp16 rounding of both sides (0.624 steps).
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) { T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc"); ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset"); return p; }
}  // namespace

int main() {
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;                                                              // several pages in a small pool
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, G = D / k::KV_Q8_GROUP;
    const int pages = 8, cells = pages * P;
    std::mt19937 rng(17);
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 5 + 3) % pages;            // a non-identity permutation
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "table");
    int8_t *kq = dalloc<int8_t>((size_t) cells * H * D), *vq = dalloc<int8_t>((size_t) cells * H * D);
    uint16_t *ks = dalloc<uint16_t>((size_t) cells * H * G), *vs = dalloc<uint16_t>((size_t) cells * H * G);
    uint16_t *kp = dalloc<uint16_t>((size_t) cells * H * D), *vp = dalloc<uint16_t>((size_t) cells * H * D);
    float *kcur = dalloc<float>(H * D), *vcur = dalloc<float>(H * D);
    int32_t* step = dalloc<int32_t>(k::kStepCount);
    std::vector<std::vector<float>> hk(cells), hv(cells);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int> positions(cells);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);
    const int n_fill = cells - 37;                                                  // leave some cells empty
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        std::vector<float> kv(H * D), vv(H * D);
        const float scale = (n % 7 == 0) ? 1e-3f : (n % 11 == 0 ? 40.f : 1.f);    // small and large groups
        for (auto& x : kv) x = nd(rng) * scale;
        for (auto& x : vv) x = nd(rng) * scale;
        if (n % 13 == 0) std::fill(kv.begin(), kv.begin() + 64, 0.f);              // an all-zero group
        hk[pos] = kv; hv[pos] = vv;
        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "step");
        ck(cudaMemcpy(kcur, kv.data(), kv.size() * 4, cudaMemcpyHostToDevice), "k");
        ck(cudaMemcpy(vcur, vv.data(), vv.size() * 4, cudaMemcpyHostToDevice), "v");
        k::kv_append_q8_step(kq, vq, ks, vs, d_table, step, kcur, vcur, s, nullptr);
        k::kv_append_step(kp, vp, d_table, step, kcur, vcur, s, nullptr);
        ck(cudaDeviceSynchronize(), "append");
    }
    // 1. codes and scales bitwise vs the host reference
    std::vector<int8_t> hkq((size_t) cells * H * D), hvq(hkq.size());
    std::vector<uint16_t> hks((size_t) cells * H * G), hvs(hks.size());
    ck(cudaMemcpy(hkq.data(), kq, hkq.size(), cudaMemcpyDeviceToHost), "d2h");
    ck(cudaMemcpy(hvq.data(), vq, hvq.size(), cudaMemcpyDeviceToHost), "d2h");
    ck(cudaMemcpy(hks.data(), ks, hks.size() * 2, cudaMemcpyDeviceToHost), "d2h");
    ck(cudaMemcpy(hvs.data(), vs, hvs.size() * 2, cudaMemcpyDeviceToHost), "d2h");
    long bad_codes = 0;
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int kvsel = 0; kvsel < 2; ++kvsel)
            for (int h = 0; h < H; ++h)
                for (int g = 0; g < G; ++g) {
                    const float* x = (kvsel ? hv[pos] : hk[pos]).data() + h * D + g * 64;
                    float amax = 0.f;
                    for (int t = 0; t < 64; ++t) amax = std::fmax(amax, std::fabs(x[t]));
                    const uint16_t sb = k::f16_from_f32(amax / 127.0f);
                    const float sf = k::f32_from_f16(sb);
                    const long long row = ((long long) table[pos / P] * H + h) * P + pos % P;
                    if ((kvsel ? hvs : hks)[row * G + g] != sb) ++bad_codes;
                    for (int t = 0; t < 64; ++t) {
                        int q = sf > 0.f ? (int) std::nearbyint(x[t] / sf) : 0;
                        q = std::clamp(q, -127, 127);
                        if ((kvsel ? hvq : hkq)[row * D + g * 64 + t] != (int8_t) q) ++bad_codes;
                    }
                }
    }
    if (bad_codes) { std::fprintf(stderr, "FAIL: %ld codes/scales differ from the host reference\n", bad_codes); ++g_fail; }
    // 2 + 3. gather random selections from both paths
    const int max_ids = 200;
    int32_t* d_ids = dalloc<int32_t>(max_ids);
    uint16_t *k8 = dalloc<uint16_t>((size_t) max_ids * H * D), *v8 = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t *k16 = dalloc<uint16_t>((size_t) max_ids * H * D), *v16 = dalloc<uint16_t>((size_t) max_ids * H * D);
    double worst = 0.0;
    for (int trial = 0; trial < 20; ++trial) {
        const int n_ids = 1 + (int) (rng() % max_ids);
        std::vector<int32_t> ids(n_ids);
        for (auto& id : ids) id = positions[rng() % n_fill];
        ck(cudaMemcpy(d_ids, ids.data(), n_ids * 4, cudaMemcpyHostToDevice), "ids");
        int32_t hstep[k::kStepCount] = {0, 0, 0, n_ids};
        ck(cudaMemcpy(step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "step");
        k::kv_gather_q8_step(kq, vq, ks, vs, d_table, d_ids, step, max_ids, s, k8, v8, nullptr);
        k::kv_gather_step(kp, vp, d_table, d_ids, step, max_ids, s, k16, v16, nullptr);
        ck(cudaDeviceSynchronize(), "gather");
        std::vector<uint16_t> a8((size_t) n_ids * H * D), b8(a8.size()), a16(a8.size()), b16(a8.size());
        ck(cudaMemcpy(a8.data(), k8, a8.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(b8.data(), v8, b8.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(a16.data(), k16, a16.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(b16.data(), v16, b16.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        for (int j = 0; j < n_ids; ++j)
            for (int h = 0; h < H; ++h) {
                const long long row = ((long long) table[ids[j] / P] * H + h) * P + ids[j] % P;
                for (int d = 0; d < D; ++d) {
                    const size_t o = ((size_t) j * H + h) * D + d;
                    for (int kvsel = 0; kvsel < 2; ++kvsel) {
                        const float sf = k::f32_from_f16((kvsel ? hvs : hks)[row * G + d / 64]);
                        const int8_t q = (kvsel ? hvq : hkq)[row * D + d];
                        const uint16_t want = k::f16_from_f32((float) q * sf);
                        const uint16_t got = (kvsel ? b8 : a8)[o];
                        if (got != want) { if (g_fail < 5) std::fprintf(stderr, "FAIL: gather id %d h %d d %d\n", j, h, d); ++g_fail; }
                        const float ref = k::f32_from_f16((kvsel ? b16 : a16)[o]);
                        const double err = std::fabs(k::f32_from_f16(got) - ref) / (sf > 0 ? sf : 1.0);
                        worst = std::max(worst, err);
                    }
                }
            }
    }
    // 4. the streamed gather: a non-resident page (page -1, as KV streaming leaves it, kv_stream.hpp) is gathered
    //    from the HOST copy's identity row (kv_append's layout); the old kernel computed a negative row and read
    //    out of bounds before the pool instead
    {
        std::vector<int8_t> cpy_kq((size_t) cells * H * D, 0), cpy_vq(cpy_kq.size(), 0);
        std::vector<uint16_t> cpy_ks((size_t) cells * H * G, 0), cpy_vs(cpy_ks.size(), 0);
        std::vector<uint16_t> pool_k((size_t) cells * H * D), pool_v(pool_k.size());
        ck(cudaMemcpy(pool_k.data(), kp, pool_k.size() * 2, cudaMemcpyDeviceToHost), "d2h f16 k pool");
        ck(cudaMemcpy(pool_v.data(), vp, pool_v.size() * 2, cudaMemcpyDeviceToHost), "d2h f16 v pool");
        std::vector<uint16_t> cpy_kp((size_t) cells * H * D, 0), cpy_vp(cpy_kp.size(), 0);
        for (int n = 0; n < n_fill; ++n) {
            const int pos = positions[n];
            for (int h = 0; h < H; ++h) {
                const long long pr = ((long long) table[pos / P] * H + h) * P + pos % P;
                const long long ir = ((long long) (pos / P) * H + h) * P + pos % P;
                for (int g = 0; g < G; ++g) { cpy_ks[ir * G + g] = hks[pr * G + g]; cpy_vs[ir * G + g] = hvs[pr * G + g]; }
                for (int d = 0; d < D; ++d) {
                    cpy_kq[ir * D + d] = hkq[pr * D + d];
                    cpy_vq[ir * D + d] = hvq[pr * D + d];
                    cpy_kp[ir * D + d] = pool_k[pr * D + d];
                    cpy_vp[ir * D + d] = pool_v[pr * D + d];
                }
            }
        }
        int8_t* d_cpy_kq = dalloc<int8_t>(cpy_kq.size());
        int8_t* d_cpy_vq = dalloc<int8_t>(cpy_vq.size());
        uint16_t* d_cpy_ks = dalloc<uint16_t>(cpy_ks.size());
        uint16_t* d_cpy_vs = dalloc<uint16_t>(cpy_vs.size());
        uint16_t* d_cpy_kp = dalloc<uint16_t>(cpy_kp.size());
        uint16_t* d_cpy_vp = dalloc<uint16_t>(cpy_vp.size());
        ck(cudaMemcpy(d_cpy_kq, cpy_kq.data(), cpy_kq.size(), cudaMemcpyHostToDevice), "h2d cpy kq");
        ck(cudaMemcpy(d_cpy_vq, cpy_vq.data(), cpy_vq.size(), cudaMemcpyHostToDevice), "h2d cpy vq");
        ck(cudaMemcpy(d_cpy_ks, cpy_ks.data(), cpy_ks.size() * 2, cudaMemcpyHostToDevice), "h2d cpy ks");
        ck(cudaMemcpy(d_cpy_vs, cpy_vs.data(), cpy_vs.size() * 2, cudaMemcpyHostToDevice), "h2d cpy vs");
        ck(cudaMemcpy(d_cpy_kp, cpy_kp.data(), cpy_kp.size() * 2, cudaMemcpyHostToDevice), "h2d cpy kp");
        ck(cudaMemcpy(d_cpy_vp, cpy_vp.data(), cpy_vp.size() * 2, cudaMemcpyHostToDevice), "h2d cpy vp");
        k::KvHostPools host_q8{}, host_f16{};
        host_q8.k_q = d_cpy_kq; host_q8.v_q = d_cpy_vq; host_q8.k_scale = d_cpy_ks; host_q8.v_scale = d_cpy_vs;
        host_f16.k_pool = d_cpy_kp; host_f16.v_pool = d_cpy_vp;
        // a page with several filled cells becomes non-resident; the selection takes its cells and resident ones
        std::vector<int> per_page(pages, 0);
        for (int n = 0; n < n_fill; ++n) ++per_page[positions[n] / P];
        int bad_page = -1;
        for (int p = 0; p < pages; ++p)
            if (per_page[p] >= 4) { bad_page = p; break; }
        std::vector<int32_t> table2 = table;
        table2[bad_page] = -1;
        int32_t* d_table2 = dalloc<int32_t>(pages);
        ck(cudaMemcpy(d_table2, table2.data(), pages * 4, cudaMemcpyHostToDevice), "table2");
        std::vector<int32_t> ids2;
        for (int n = 0; n < n_fill && ids2.size() < 6; ++n)
            if (positions[n] / P == bad_page) ids2.push_back(positions[n]);
        for (int n = 0; n < n_fill && ids2.size() < 12; ++n)
            if (positions[n] / P != bad_page) ids2.push_back(positions[n]);
        std::sort(ids2.begin(), ids2.end());
        const int n2 = (int) ids2.size();
        ck(cudaMemcpy(d_ids, ids2.data(), n2 * 4, cudaMemcpyHostToDevice), "ids2");
        int32_t hstep2[k::kStepCount] = {0, 0, 0, n2};
        ck(cudaMemcpy(step, hstep2, sizeof hstep2, cudaMemcpyHostToDevice), "step2");
        k::kv_gather_q8_step(kq, vq, ks, vs, d_table2, d_ids, step, max_ids, s, k8, v8, nullptr, &host_q8);
        k::kv_gather_step(kp, vp, d_table2, d_ids, step, max_ids, s, k16, v16, nullptr, &host_f16);
        ck(cudaDeviceSynchronize(), "streamed gather");
        std::vector<uint16_t> a8((size_t) n2 * H * D), b8(a8.size()), a16(a8.size()), b16(a8.size());
        ck(cudaMemcpy(a8.data(), k8, a8.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(b8.data(), v8, b8.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(a16.data(), k16, a16.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(b16.data(), v16, b16.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        long bad = 0;
        for (int j = 0; j < n2; ++j)
            for (int h = 0; h < H; ++h) {
                const long long ir = ((long long) (ids2[j] / P) * H + h) * P + ids2[j] % P;
                for (int d = 0; d < D; ++d) {
                    const size_t o = ((size_t) j * H + h) * D + d;
                    for (int kvsel = 0; kvsel < 2; ++kvsel) {
                        const float sf = k::f32_from_f16((kvsel ? cpy_vs : cpy_ks)[ir * G + d / 64]);
                        const int8_t q = (kvsel ? cpy_vq : cpy_kq)[ir * D + d];
                        const uint16_t want = k::f16_from_f32((float) q * sf);
                        if ((kvsel ? b8 : a8)[o] != want) ++bad;
                        if ((kvsel ? b16 : a16)[o] != (kvsel ? cpy_vp : cpy_kp)[ir * D + d]) ++bad;
                    }
                }
            }
        if (bad) { std::fprintf(stderr, "FAIL: %ld streamed-gather values differ from the host copy\n", bad); ++g_fail; }
        else {
            const int n_miss = (int) std::count_if(ids2.begin(), ids2.end(), [&](int c) { return c / P == bad_page; });
            std::printf("streamed gather: %d non-resident + %d resident cells, int8 and fp16 (OK)\n", n_miss, n2 - n_miss);
        }
        cudaFree(d_cpy_kq); cudaFree(d_cpy_vq); cudaFree(d_cpy_ks); cudaFree(d_cpy_vs);
        cudaFree(d_cpy_kp); cudaFree(d_cpy_vp); cudaFree(d_table2);
    }
    // |x - q*sf| <= sf/2 (the stored scale keeps |x|/sf <= 127.06, so no clamping), plus the fp16 rounding of each
    // side: at most 2^-11 relative of a value up to 127*sf, i.e. 127/2048 = 0.062 steps each -> 0.5 + 0.124
    if (!(worst <= 0.5 + 2.0 * 127.0 / 2048.0 + 1e-3)) { std::fprintf(stderr, "FAIL: INT8 vs FP16 error %.3f steps\n", worst); ++g_fail; }
    std::printf("kv_q8_parity: %s (worst INT8-vs-FP16 error %.3f quantization steps)\n", g_fail ? "FAILED" : "OK", worst);
    return g_fail ? 1 : 0;
}
