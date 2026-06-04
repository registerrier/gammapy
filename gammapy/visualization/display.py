import io
import base64

import matplotlib.pyplot as plt


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


class FitResultDisplay:
    def __init__(self, result):
        self.result = result

    def __repr__(self):
        return str(self.result)

    def _make_parameter_table(self):
        table = self.result.models.to_parameters_table()
        return table.to_pandas()

    def _render_parameter_table(self, df):
        if "frozen" not in df.columns:
            return df.to_html(index=False, border=0)

        def _style_frozen(row):
            color = "color: grey;" if row["frozen"] else ""
            return [color] * len(row)

        return df.style.apply(_style_frozen, axis=1).to_html()

    def _make_covariance_plot(self):
        if self.result.covariance_result is None:
            return "<p>No covariance available</p>"

        try:
            ax = self.result.models.covariance.plot_correlation()
            img = fig_to_base64(ax.figure)
            return f'<img src="data:image/png;base64,{img}"/>'
        except Exception as e:
            return f"<p>Failed to generate covariance plot: {e}</p>"

    def _repr_html_(self):
        r = self.result
        out = f"""
        <h3>Fit Result</h3>
        <ul>
            <li><b>Success:</b> {r.success}</li>
            <li><b>Total stat:</b> {r.total_stat:.3f}</li>
            <li><b>Message:</b> {r.message}</li>
            <li><b>nfev:</b> {r.nfev}</li>
            <li><b>Backend:</b> {r.backend}</li>
            <li><b>Method:</b> {r.method}</li>
        </ul>
        """

        iminuit_obj = getattr(r, "minuit", None)
        iminuit_html = iminuit_obj._repr_html_() if iminuit_obj else "Not available"

        out += f"""
        <details>
            <summary><b>iminuit details</b></summary>
            <div>{iminuit_html}</div>
        </details>
        """

        try:
            df = self._make_parameter_table()
            table_html = self._render_parameter_table(df)
        except Exception as e:
            table_html = f"<p>Failed to build parameter table: {e}</p>"

        out += f"""
        <details>
            <summary><b>Parameters</b></summary>
            {table_html}
        </details>
        """

        cov_html = self._make_covariance_plot()

        out += f"""
        <details>
            <summary><b>Covariance / Correlation</b></summary>
            {cov_html}
        </details>
        """

        return out
