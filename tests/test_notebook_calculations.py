"""Small numerical regressions for functions defined in the main notebook."""

import ast
import contextlib
import datetime
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, linregress


ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = {
    'np': np, 'pd': pd, 'datetime': datetime,
    'linregress': linregress, 'chi2_contingency': chi2_contingency,
}
# Load the notebook's actual functions without executing plots or the full analysis.
notebook = json.loads((ROOT / 'DBA_project_ForecastabilityIndex.ipynb').read_text())
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<notebook functions>', 'exec'), NAMESPACE)


class NotebookCalculationsTests(unittest.TestCase):
    def test_store_dates_preserve_active_and_remove_closed_sales(self):
        sales = pd.DataFrame({
            'store_location_rk': [1, 1, 2, 2, 3],
            'period_start_dt': pd.to_datetime(['2018-03-14', '2018-03-19', '2018-03-19', '2018-03-26', '2018-03-19']),
            'demand': [10, 20, 30, 40, 50],
        })
        stores = pd.DataFrame({
            'STORE_LOCATION_RK': [1, 2, 3],
            'STORE_OPEN_DTTM': ['15Mar2018:14:08:08'] * 3,
            'STORE_CLOSURE_DTTM': ['01Jan5999:00:00:00', '20Mar2018:00:00:00', None],
        })
        result = NAMESPACE['filter_open_stores'](sales, stores)
        self.assertEqual(result['demand'].tolist(), [20, 30, 50])

    def test_unknown_store_raises_instead_of_dropping_sales(self):
        sales = pd.DataFrame({'store_location_rk': [999], 'period_start_dt': pd.to_datetime(['2020-01-01'])})
        stores = pd.DataFrame({
            'STORE_LOCATION_RK': [1], 'STORE_OPEN_DTTM': ['01Jan2018:00:00:00'],
            'STORE_CLOSURE_DTTM': ['01Jan5999:00:00:00'],
        })
        with self.assertRaises(ValueError):
            NAMESPACE['filter_open_stores'](sales, stores)

    def test_forecast_first_step_is_future_and_last_window_is_included(self):
        sales = pd.DataFrame({
            'product_rk': [1] * 4, 'store_location_rk': [1] * 4,
            'period_start_dt': pd.date_range('2020-01-06', periods=4, freq='7D'),
            'demand': [10., 20., 30., 40.],
        })
        with contextlib.redirect_stdout(io.StringIO()):
            forecasts = NAMESPACE['build_naive_forecasts_for_all'](sales, horizon=2)
        self.assertEqual(forecasts['naive_forecast'].tolist(), [10., 10., 20., 20.])
        self.assertEqual(forecasts['actual_demand'].tolist(), [20., 30., 30., 40.])
        self.assertTrue((forecasts['forecast_date'] > forecasts['train_date']).all())
        self.assertEqual(forecasts['forecast_date'].max(), sales['period_start_dt'].max())

    def test_missing_week_does_not_shorten_forecast_horizon(self):
        sales = pd.DataFrame({
            'product_rk': [1] * 3, 'store_location_rk': [1] * 3,
            'period_start_dt': pd.to_datetime(['2020-01-06', '2020-01-20', '2020-01-27']),
            'demand': [10., 30., 40.],
        })
        with contextlib.redirect_stdout(io.StringIO()):
            forecasts = NAMESPACE['build_naive_forecasts_for_all'](sales, horizon=2)
        self.assertEqual(forecasts['forecast_date'].tolist(), list(pd.to_datetime(['2020-01-13', '2020-01-20'])))
        self.assertTrue(pd.isna(forecasts['actual_demand'].iloc[0]))
        self.assertEqual(forecasts['actual_demand'].iloc[1], 30.)

    def test_short_history_and_unknown_future_are_not_scored(self):
        sales = pd.DataFrame({
            'product_rk': [1] * 3, 'store_location_rk': [1] * 3,
            'period_start_dt': pd.date_range('2020-01-06', periods=3, freq='7D'),
            'demand': [10., 20., np.nan],
        })
        with contextlib.redirect_stdout(io.StringIO()):
            forecasts = NAMESPACE['build_naive_forecasts_for_all'](sales, horizon=2)
        self.assertTrue(forecasts.empty)
        self.assertIn('actual_demand', forecasts.columns)
        with self.assertRaises(ValueError):
            NAMESPACE['build_naive_forecasts_for_all'](sales, horizon=0)

    def test_wape_uses_observed_ratio_and_bootstrap_is_repeatable(self):
        actual = np.array([1., 2., 4., 8., 16., 32.])
        predicted = np.array([0., 1., 4., 4., 12., 20.])
        function = NAMESPACE['calculate_bootstrap_wape']
        first = function(actual, predicted)
        second = function(actual, predicted)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 22. / 63.)
        self.assertLessEqual(first[1], first[2])

    def test_wape_ignores_missing_pairs_and_handles_zero_demand(self):
        function = NAMESPACE['calculate_bootstrap_wape']
        self.assertAlmostEqual(function([10., np.nan, 30.], [5., 99., np.nan])[0], .5)
        self.assertTrue(np.isnan(function([0., 0.], [1., 1.])[0]))
        self.assertEqual(function([10., 20.], [10., 20.]), (0., 0., 0.))

    def test_zero_wape_belongs_to_easy_cluster(self):
        # Use the exact bins and cut call from the notebook's final cell.
        source = ''.join(notebook['cells'][85]['source'])
        prefix = source[:source.index('cluster_stats =')]
        state = {'np': np, 'pd': pd, 'fi_table': pd.DataFrame({'WAPE': [0., .5, .75, 2.]})}
        exec(prefix, state)
        self.assertEqual(state['fi_cluster']['Forecast_Cluster'].cat.codes.tolist(), [0, 1, 2, 2])

    def test_segment_counts_store_product_pairs_and_sales_without_window_duplicates(self):
        features = pd.DataFrame({
            'store_location_rk': [1, 2], 'product_rk': [100, 100],
            'volume_group': ['A', 'A'], 'variability_group': ['C', 'C'],
            'has_seasonality': [0, 0], 'has_trend': [0, 0], 'is_intermittent': [0, 0],
        })
        forecasts = pd.DataFrame({
            'store_location_rk': [1, 1, 2, 2], 'product_rk': [100] * 4,
            'forecast_date': pd.to_datetime(['2020-01-13'] * 4),
            'actual_demand': [10., 10., 20., 20.],
            'naive_forecast': [5., 5., 15., 15.],
        })
        state = {
            'np': np, 'pd': pd, 'RANDOM_SEED': 42,
            'ts_id_columns': ['store_location_rk', 'product_rk'],
            'ts_retail': features, 'forecasts_df': forecasts,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(notebook['cells'][77]['source']), state)
        row = state['fi_table'].iloc[0]
        self.assertEqual(row['n_series'], 2)
        self.assertEqual(row['total_volume'], 30.)
        self.assertAlmostEqual(row['WAPE'], 1. / 3.)

    def test_constant_and_short_series_have_no_seasonality_or_trend(self):
        with np.errstate(all='raise'), contextlib.redirect_stdout(io.StringIO()):
            for values in [np.zeros(30), np.ones(30), np.array([1., 2.]), np.array([np.nan])]:
                self.assertEqual(NAMESPACE['has_seasonality'](values), 0)
                self.assertEqual(NAMESPACE['get_trend_flag'](values), 0)


if __name__ == '__main__':
    unittest.main()
