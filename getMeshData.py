from __future__ import annotations

from typing import Any

import igl
import numpy as np
import scipy.sparse as sp


def _as_zero_based_triangles(T: np.ndarray, nv: int) -> np.ndarray:
    tri = np.asarray(T, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("T must have shape (m, 3).")
    if tri.size > 0 and tri.min() == 1 and tri.max() <= nv:
        tri = tri - 1
    if tri.size > 0 and (tri.min() < 0 or tri.max() >= nv):
        raise ValueError("Triangle indices are out of range.")
    return tri


def _row_normalize(v: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    out = np.zeros_like(v)
    nz = nrm[:, 0] > 0.0
    out[nz] = v[nz] / nrm[nz]
    return out


def _edge_topology_from_igl(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    num_triangles = triangles.shape[0]
    _, unique_edges_raw, edge_map_raw, unique_to_halfedges = igl.unique_edge_map_lists(triangles)

    # Match MATLAB: unique(sort(edges,2),'rows')
    unique_edges_sorted = np.sort(np.asarray(unique_edges_raw, dtype=np.int64), axis=1)

    edge_map_raw_i64 = np.asarray(edge_map_raw, dtype=np.int64).reshape(-1)
    counts = np.bincount(edge_map_raw_i64, minlength=unique_edges_sorted.shape[0])
    if np.any(counts > 2):
        raise ValueError("Non manifold triangles found.")

    lex = np.lexsort((unique_edges_sorted[:, 1], unique_edges_sorted[:, 0]))
    is_boundary_lex = counts[lex] == 1

    # Match MATLAB: sort(isBoundaryEdge) => interior first, boundary last
    order = np.concatenate([lex[~is_boundary_lex], lex[is_boundary_lex]])
    edges = unique_edges_sorted[order]

    num_edges = edges.shape[0]
    num_interior_edges = int(np.sum(~is_boundary_lex))
    num_boundary_edges = num_edges - num_interior_edges

    old_to_new = np.empty(num_edges, dtype=np.int64)
    old_to_new[order] = np.arange(num_edges, dtype=np.int64)
    edge_map = old_to_new[edge_map_raw_i64]

    # MATLAB-style mapping from directed half-edge list to per-triangle 3 edges.
    triangles2edges = np.sort(edge_map.reshape((num_triangles, 3), order="F"), axis=1)

    edges2triangles = np.full((num_edges, 2), -1, dtype=np.int64)
    order_i64 = np.asarray(order, dtype=np.int64).reshape(-1)
    for new_e in range(num_edges):
        old_e = int(order_i64[new_e])
        half_ids = unique_to_halfedges[old_e]
        # Libigl documents half-edge index order as f + #F*c.
        tris = sorted({int(h % num_triangles) for h in half_ids})
        if len(tris) >= 1:
            edges2triangles[new_e, 0] = tris[0]
        if len(tris) >= 2:
            edges2triangles[new_e, 1] = tris[1]

    is_boundary_edge = np.zeros(num_edges, dtype=bool)
    if num_boundary_edges > 0:
        is_boundary_edge[num_interior_edges:] = True

    return edges, triangles2edges, edges2triangles, num_interior_edges, num_boundary_edges


def get_mesh_data(X: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    vertices = np.asarray(X, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("X must have shape (n, 3).")

    triangles = _as_zero_based_triangles(T, vertices.shape[0])
    num_vertices = vertices.shape[0]
    num_triangles = triangles.shape[0]

    v1 = vertices[triangles[:, 0], :]
    v2 = vertices[triangles[:, 1], :]
    v3 = vertices[triangles[:, 2], :]

    normalf = np.cross(v2 - v1, v3 - v1)
    face_normals = _row_normalize(normalf)

    edges, triangles2edges, edges2triangles, num_interior_edges, num_boundary_edges = _edge_topology_from_igl(triangles)
    num_edges = edges.shape[0]

    is_boundary_edge = np.zeros(num_edges, dtype=bool)
    if num_boundary_edges > 0:
        is_boundary_edge[num_interior_edges:] = True

    triangle_barycenters = np.asarray(igl.barycenter(vertices, triangles), dtype=np.float64)

    A = np.linalg.norm(v1 - v2, axis=1)
    B = np.linalg.norm(v2 - v3, axis=1)
    C = np.linalg.norm(v3 - v1, axis=1)
    S = 0.5 * (A + B + C)
    triangle_areas = np.sqrt(np.clip(S * (S - A) * (S - B) * (S - C), 0.0, None))
    triangle_edge_b_length = B

    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    if num_interior_edges > 0:
        t0 = edges2triangles[:num_interior_edges, 0]
        t1 = edges2triangles[:num_interior_edges, 1]
        dual_edge_lengths = np.linalg.norm(triangle_barycenters[t0] - triangle_barycenters[t1], axis=1)
        primal_over_dual_weight = np.divide(
            edge_lengths[:num_interior_edges],
            dual_edge_lengths,
            out=np.zeros(num_interior_edges, dtype=np.float64),
            where=dual_edge_lengths > 0.0,
        )
    else:
        dual_edge_lengths = np.zeros(0, dtype=np.float64)
        primal_over_dual_weight = np.zeros(0, dtype=np.float64)

    rows = np.concatenate(
        [
            np.arange(num_edges, dtype=np.int64),
            np.arange(num_edges, dtype=np.int64),
        ]
    )
    cols = np.concatenate([edges[:, 0], edges[:, 1]])
    vals = np.concatenate([np.ones(num_edges), -np.ones(num_edges)]).astype(np.float64)
    primal_incidence = sp.csr_matrix((vals, (rows, cols)), shape=(num_edges, num_vertices), dtype=np.float64)

    if num_interior_edges > 0:
        ei = np.arange(num_interior_edges, dtype=np.int64)
        tri_a = edges2triangles[:num_interior_edges, 0]
        tri_b = edges2triangles[:num_interior_edges, 1]
        incidence = sp.csr_matrix(
            (
                np.concatenate([np.ones(num_interior_edges), -np.ones(num_interior_edges)]),
                (np.concatenate([ei, ei]), np.concatenate([tri_a, tri_b])),
            ),
            shape=(num_interior_edges, num_triangles),
            dtype=np.float64,
        )
    else:
        incidence = sp.csr_matrix((0, num_triangles), dtype=np.float64)

    dual_graph_l = incidence.T @ incidence

    t_rows = np.repeat(np.arange(num_triangles, dtype=np.int64), 3)
    t_cols = triangles.reshape(-1)
    triangle2verts = sp.csr_matrix(
        (np.ones(t_rows.size), (t_rows, t_cols)),
        shape=(num_triangles, num_vertices),
        dtype=np.float64,
    )

    vert_normals = triangle2verts.T @ (face_normals * triangle_areas[:, None])
    vert_normals = _row_normalize(np.asarray(vert_normals, dtype=np.float64))

    edge_weights = np.divide(
        1.0,
        primal_over_dual_weight,
        out=np.zeros_like(primal_over_dual_weight),
        where=primal_over_dual_weight > 0.0,
    )

    adj = sp.csr_matrix(
        (
            np.ones(3 * num_triangles, dtype=np.float64),
            (
                np.repeat(np.arange(num_triangles, dtype=np.int64), 3),
                triangles2edges.reshape(-1),
            ),
        ),
        shape=(num_triangles, num_edges),
        dtype=np.float64,
    )

    adj_trilabeled = sp.csr_matrix(
        (
            np.repeat(np.arange(num_triangles, dtype=np.float64) + 1.0, 3),
            (
                np.repeat(np.arange(num_triangles, dtype=np.int64), 3),
                triangles2edges.reshape(-1),
            ),
        ),
        shape=(num_triangles, num_edges),
        dtype=np.float64,
    )

    edge_xedge2tri = (adj.T @ adj_trilabeled).tocsr()
    edge_xedge2tri.setdiag(0.0)
    edge_xedge2tri.eliminate_zeros()

    if num_interior_edges > 0:
        i0 = edges2triangles[:num_interior_edges, 0]
        i1 = edges2triangles[:num_interior_edges, 1]
        eidx = np.arange(num_interior_edges, dtype=np.float64)
        tri_xtri2edge = sp.csr_matrix(
            (eidx + 1.0, (i0, i1)),
            shape=(num_triangles, num_triangles),
            dtype=np.float64,
        )
        tri_xtri2edge = tri_xtri2edge + tri_xtri2edge.T

        tri_xedge2tri = sp.csr_matrix(
            (i1 + 1.0, (i0, np.arange(num_interior_edges))),
            shape=(num_triangles, num_edges),
            dtype=np.float64,
        )
        tri_xedge2tri = tri_xedge2tri + sp.csr_matrix(
            (i0 + 1.0, (i1, np.arange(num_interior_edges))),
            shape=(num_triangles, num_edges),
            dtype=np.float64,
        )
    else:
        tri_xtri2edge = sp.csr_matrix((num_triangles, num_triangles), dtype=np.float64)
        tri_xedge2tri = sp.csr_matrix((num_triangles, num_edges), dtype=np.float64)

    data: dict[str, Any] = {
        "vertices": vertices,
        "triangles": triangles,
        "numVertices": num_vertices,
        "numTriangles": num_triangles,
        "faceNormals": face_normals,
        "edges": edges,
        "numEdges": num_edges,
        "numInteriorEdges": num_interior_edges,
        "numBoundaryEdges": num_boundary_edges,
        "triangles2edges": triangles2edges,
        "edges2triangles": edges2triangles,
        "isBoundaryEdge": is_boundary_edge,
        "triangleBarycenters": triangle_barycenters,
        "triangleAreas": triangle_areas,
        "triangleEdgeBLength": triangle_edge_b_length,
        "edgeLengths": edge_lengths,
        "dualEdgeLengths": dual_edge_lengths,
        "primalOverDualWeight": primal_over_dual_weight,
        "primalIncidence": primal_incidence,
        "incidenceMatrix": incidence,
        "dualGraphL": dual_graph_l,
        "triangle2verts": triangle2verts,
        "vertNormals": vert_normals,
        "edgeWeights": edge_weights,
        "edgeXedge2tri": edge_xedge2tri,
        "triXtri2edge": tri_xtri2edge,
        "triXedge2tri": tri_xedge2tri,
    }
    return data


def getMeshData(X: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    return get_mesh_data(X, T)
