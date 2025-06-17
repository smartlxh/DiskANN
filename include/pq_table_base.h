// pq_table_base.h
#pragma once
#include <cstdint>
#include <vector>

class PQTableBase {
  public:
    virtual ~PQTableBase() {};

    // 查询预处理（中心化/旋转）
    virtual void preprocess_query(float* query);

    virtual void train(size_t n, const float* x);

    // 计算每个chunk的距离表
    virtual void populate_chunk_distances(const float* query, float* out_dists);

    // 获取chunk数量 pq
    virtual uint64_t get_num_chunks();

    // 支持OPQ等扩展功能的虚函数
    virtual void apply_rotation(float* vec) const { /* 默认无旋转 */ }

    virtual void compute_dists(const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                               uint8_t *data, uint8_t *pq_coord_scratch, float* pq_dists);

    virtual uint64_t get_num_points();

    virtual void load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data);

#ifdef EXEC_ENV_OLS
    virtual void load_pq_centroid_bin(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks) {};
#else
    virtual void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks) {};
#endif
};
