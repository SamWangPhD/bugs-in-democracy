import React, { useEffect, useMemo, useState, createContext, useContext } from "react";
import Papa from "papaparse";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Download, Filter, BarChart2, ScatterChart as ScatterIcon, Table as TableIcon } from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip as RTooltip,
  CartesianGrid,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  Legend,
} from "recharts";

/**
 * ==============================================
 * Ballot Analysis Dashboard (no uploads needed)
 * ==============================================
 *
 * Place `election_table.csv` in the same directory
 * as this app (or adjust CSV_PATH below). The app
 * fetches it directly from your laptop (served as
 * a static file) and provides filters + charts.
 *
 * Columns expected (as provided by user):
 * filename, null_gamma, null_gamma_recalculated, complete_gamma, gamma,
 * choices, candidates, level, partisan, type, irv_winner, condorcet_winner,
 * plurality_winner, approval_winner, median_voter_preference,
 * median_voter_preference_position, median_voter_position,
 * irv_median_voter_distance, condorcet_median_voter_distance,
 * median_voter_preference_distance, bimodality, partisan2, diff,
 * linearvotersfailure, mirror
 */

// ---- CONFIG ----
const CSV_PATH = "./election_table.csv"; // adjust if needed

// ---- Types / Context ----
const DataCtx = createContext(null);
const useData = () => {
  const ctx = useContext(DataCtx);
  if (!ctx) throw new Error("useData must be used within provider");
  return ctx;
};

export default function App() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Filters
  const [selLevels, setSelLevels] = useState(new Set());
  const [selTypes, setSelTypes] = useState(new Set());
  const [selPartisan, setSelPartisan] = useState(new Set()); // values: "YES","No","DP","RP"
  const [selK, setSelK] = useState(new Set()); // candidates
  const [selChoices, setSelChoices] = useState(new Set());
  const [search, setSearch] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const text = await fetch(CSV_PATH).then(r => {
          if (!r.ok) throw new Error(`Failed to fetch ${CSV_PATH}`);
          return r.text();
        });
        const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
        const data = parsed.data.map((r) => sanitizeRow(r));
        setRows(data);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const domain = useMemo(() => buildDomain(rows), [rows]);

  const filtered = useMemo(() => {
    let arr = rows;
    if (selLevels.size) arr = arr.filter(r => selLevels.has(r.level));
    if (selTypes.size) arr = arr.filter(r => selTypes.has(r.type));
    if (selPartisan.size) arr = arr.filter(r => selPartisan.has(r.partisan));
    if (selK.size) arr = arr.filter(r => selK.has(r.candidates));
    if (selChoices.size) arr = arr.filter(r => selChoices.has(r.choices));
    const q = search.trim().toLowerCase();
    if (q) arr = arr.filter(r => Object.values(r).some(v => String(v).toLowerCase().includes(q)));
    return arr;
  }, [rows, selLevels, selTypes, selPartisan, selK, selChoices, search]);

  const ctxValue = useMemo(() => ({
    loading, error, rows, filtered, domain,
    selLevels, setSelLevels,
    selTypes, setSelTypes,
    selPartisan, setSelPartisan,
    selK, setSelK,
    selChoices, setSelChoices,
    search, setSearch,
  }), [loading, error, rows, filtered, domain, selLevels, selTypes, selPartisan, selK, selChoices, search]);

  return (
    <DataCtx.Provider value={ctxValue}>
      <div className="min-h-screen bg-neutral-50 p-6">
        <div className="mx-auto max-w-7xl space-y-6">
          <Header />
          {error ? <ErrorMsg msg={error} /> : null}
          <MainTabs />
        </div>
      </div>
    </DataCtx.Provider>
  );
}

function Header() {
  const { rows, filtered } = useData();
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Ballot Analysis Dashboard</h1>
        <p className="text-sm text-neutral-600">Loaded rows: <strong>{rows.length}</strong> · Showing: <strong>{filtered.length}</strong></p>
      </div>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return (
    <Card className="border-red-300 bg-red-50">
      <CardHeader>
        <CardTitle className="text-red-700">Error loading CSV</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-red-700 whitespace-pre-wrap">{msg}</p>
      </CardContent>
    </Card>
  );
}

function MainTabs() {
  const { loading, filtered } = useData();
  return (
    <Tabs defaultValue="overview">
      <TabsList>
        <TabsTrigger value="overview"><TableIcon className="mr-2 h-4 w-4"/>Overview</TabsTrigger>
        <TabsTrigger value="charts" disabled={loading || filtered.length===0}><BarChart2 className="mr-2 h-4 w-4"/>Charts</TabsTrigger>
      </TabsList>
      <TabsContent value="overview">
        <Filters />
        <OverviewTable />
      </TabsContent>
      <TabsContent value="charts">
        <Filters compact />
        <ChartsPanel />
      </TabsContent>
    </Tabs>
  );
}

// ----------- Filters -----------
function Filters({ compact=false }) {
  const { domain, selLevels, setSelLevels, selTypes, setSelTypes, selPartisan, setSelPartisan, selK, setSelK, selChoices, setSelChoices, search, setSearch, filtered } = useData();

  function toggle(set, curSet, val) {
    const next = new Set(curSet);
    if (next.has(val)) next.delete(val); else next.add(val);
    set(next);
  }

  return (
    <Card className="shadow-sm mb-4">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center gap-2"><Filter className="h-4 w-4"/>Filters <Badge variant="secondary">{filtered.length}</Badge></CardTitle>
      </CardHeader>
      <CardContent>
        <div className={`grid gap-4 ${compact ? "md:grid-cols-4" : "md:grid-cols-5"}`}>
          <FilterGroup label="Candidates (k)" values={domain.k} selected={selK} onToggle={(v)=>toggle(setSelK, selK, v)} />
          <FilterGroup label="Choices" values={domain.choices} selected={selChoices} onToggle={(v)=>toggle(setSelChoices, selChoices, v)} />
          <FilterGroup label="Level" values={domain.level} selected={selLevels} onToggle={(v)=>toggle(setSelLevels, selLevels, v)} />
          <FilterGroup label="Type" values={domain.type} selected={selTypes} onToggle={(v)=>toggle(setSelTypes, selTypes, v)} />
          <FilterGroup label="Partisan" values={domain.partisan} selected={selPartisan} onToggle={(v)=>toggle(setSelPartisan, selPartisan, v)} />
        </div>
        <Separator className="my-3" />
        <div className="flex items-center gap-3">
          <Input className="max-w-md" placeholder="Search all columns…" value={search} onChange={e=>setSearch(e.target.value)} />
          <Button variant="secondary" onClick={()=>{ setSelLevels(new Set()); setSelTypes(new Set()); setSelPartisan(new Set()); setSelK(new Set()); setSelChoices(new Set()); setSearch(""); }}>Clear</Button>
          <ExportButton />
        </div>
      </CardContent>
    </Card>
  );
}

function FilterGroup({ label, values, selected, onToggle }) {
  return (
    <div>
      <Label className="mb-2 block text-xs uppercase tracking-wide text-neutral-500">{label}</Label>
      <div className="flex flex-wrap gap-2">
        {values.map(v => (
          <button key={String(v)} onClick={()=>onToggle(v)} className={`rounded-full px-3 py-1 text-sm border ${selected.has(v)?"bg-black text-white border-black":"bg-white hover:bg-neutral-100"}`}>
            {String(v)}
          </button>
        ))}
      </div>
    </div>
  );
}

function ExportButton() {
  const { filtered } = useData();
  function toCSV(rows) {
    if (!rows.length) return "";
    const cols = Object.keys(rows[0]);
    const header = cols.join(",");
    const body = rows.map(r => cols.map(c => JSON.stringify(r[c] ?? "")).join(",")).join("
");
    return header + "
" + body;
  }
  return (
    <Button onClick={() => {
      const blob = new Blob([toCSV(filtered)], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = "filtered_election_table.csv"; a.click();
      URL.revokeObjectURL(url);
    }}><Download className="mr-2 h-4 w-4"/>Export filtered CSV</Button>
  );
}

// ----------- Overview Table -----------
function OverviewTable() {
  const { filtered } = useData();
  const cols = [
    "filename","gamma","null_gamma","null_gamma_recalculated","complete_gamma","choices","candidates","level","partisan","type","irv_winner","condorcet_winner","plurality_winner","approval_winner","median_voter_preference","median_voter_preference_position","median_voter_position","irv_median_voter_distance","condorcet_median_voter_distance","median_voter_preference_distance","bimodality","partisan2","diff","linearvotersfailure","mirror"
  ];
  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Elections</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="w-full overflow-auto rounded-lg border">
          <table className="w-full text-sm">
            <thead className="bg-neutral-100">
              <tr>
                {cols.map(c => <th key={c} className="px-3 py-2 text-left font-semibold">{c}</th>)}
              </tr>
            </thead>
            <tbody>
              {filtered.map((r,i)=> (
                <tr key={i} className="hover:bg-neutral-50">
                  {cols.map(c => <td key={c} className="px-3 py-2 border-t">{String(r[c] ?? "")}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

// ----------- Charts Panel -----------
function ChartsPanel() {
  const { filtered } = useData();
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card className="shadow-sm">
        <CardHeader><CardTitle className="text-base">gamma vs candidates (k)</CardTitle></CardHeader>
        <CardContent><ScatterXY rows={filtered} xKey="candidates" yKey="gamma" xLabel="k" yLabel="gamma"/></CardContent>
      </Card>

      <Card className="shadow-sm">
        <CardHeader><CardTitle className="text-base">gamma vs null_gamma</CardTitle></CardHeader>
        <CardContent><ScatterXY rows={filtered} xKey="null_gamma" yKey="gamma" xLabel="null_gamma" yLabel="gamma"/></CardContent>
      </Card>

      <Card className="shadow-sm">
        <CardHeader><CardTitle className="text-base">Distribution of gamma</CardTitle></CardHeader>
        <CardContent><Histogram rows={filtered} valueKey="gamma" bins={20} /></CardContent>
      </Card>

      <Card className="shadow-sm">
        <CardHeader><CardTitle className="text-base">Winner–Median distances</CardTitle></CardHeader>
        <CardContent>
          <BarPairs rows={filtered} pairs={[{ key:"irv_median_voter_distance", label:"IRV" }, { key:"condorcet_median_voter_distance", label:"Condorcet" }, { key:"median_voter_preference_distance", label:"Plurality/Pref" }]} />
        </CardContent>
      </Card>

      <Card className="shadow-sm">
        <CardHeader><CardTitle className="text-base">Bimodality vs candidates</CardTitle></CardHeader>
        <CardContent><ScatterXY rows={filtered} xKey="candidates" yKey="bimodality" xLabel="k" yLabel="bimodality"/></CardContent>
      </Card>

      <Card className="shadow-sm">
        <CardHeader><CardTitle className="text-base">Counts by partisan type</CardTitle></CardHeader>
        <CardContent><CountsBy rows={filtered} groupKey="partisan" /></CardContent>
      </Card>
    </div>
  );
}

// ----------- Chart helpers -----------
function ScatterXY({ rows, xKey, yKey, xLabel, yLabel }) {
  const data = rows.map(r => ({ x: num(r[xKey]), y: num(r[yKey]), name: r.filename })).filter(d => Number.isFinite(d.x) && Number.isFinite(d.y));
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey="x" name={xLabel} label={{ value: xLabel, position: "insideBottom", dy: 10 }} />
          <YAxis type="number" dataKey="y" name={yLabel} label={{ value: yLabel, angle: -90, position: "insideLeft" }} />
          <RTooltip formatter={(v, n, p) => [v, n]} labelFormatter={(l)=>`${xLabel}=${l}`} />
          <Scatter data={data} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function Histogram({ rows, valueKey, bins=20 }) {
  const vals = rows.map(r => num(r[valueKey])).filter(Number.isFinite);
  if (!vals.length) return <p className="text-sm text-neutral-500">No numeric data</p>;
  const min = Math.min(...vals), max = Math.max(...vals);
  const width = (max - min) / (bins || 20) || 1;
  const edges = Array.from({ length: bins+1 }, (_, i) => min + i*width);
  const counts = Array.from({ length: bins }, () => 0);
  vals.forEach(v => {
    let idx = Math.floor((v - min) / width);
    if (idx >= bins) idx = bins-1;
    if (idx < 0) idx = 0;
    counts[idx]++;
  });
  const data = counts.map((c,i)=>({ bin: Number((edges[i]).toFixed(3)), count: c }));
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="bin" tickFormatter={(v)=>String(v)} label={{ value: valueKey, position: "insideBottom", dy: 10 }} />
          <YAxis />
          <RTooltip />
          <Bar dataKey="count" name="count" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function BarPairs({ rows, pairs }) {
  const data = rows.map(r => {
    const o = { name: r.filename };
    pairs.forEach(p => o[p.label] = num(r[p.key]));
    return o;
  });
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data.slice(0, 50)} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" hide />
          <YAxis />
          <RTooltip />
          <Legend />
          {pairs.map(p => <Bar key={p.label} dataKey={p.label} />)}
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-neutral-500">Showing first 50 elections. Use export to analyze all.</p>
    </div>
  );
}

function CountsBy({ rows, groupKey }) {
  const map = new Map();
  rows.forEach(r => {
    const k = String(r[groupKey] ?? "").trim() || "(missing)";
    map.set(k, (map.get(k) || 0) + 1);
  });
  const data = Array.from(map.entries()).map(([k,v])=>({ name: k, count: v }));
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis allowDecimals={false} />
          <RTooltip />
          <Bar dataKey="count" name="count" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ----------- Helpers -----------
function num(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : NaN;
}

function sanitizeRow(r) {
  const out = { ...r };
  // Coerce key numeric fields used in charts/filters
  [
    "gamma","null_gamma","null_gamma_recalculated","complete_gamma","choices","candidates",
    "irv_median_voter_distance","condorcet_median_voter_distance","median_voter_preference_distance",
    "bimodality","diff","linearvotersfailure","mirror"
  ].forEach(k => { if (k in out) out[k] = num(out[k]); });
  // Strings to trimmed
  ["filename","level","partisan","type","irv_winner","condorcet_winner","plurality_winner","approval_winner","partisan2"].forEach(k=>{ if (k in out && out[k]!=null) out[k] = String(out[k]).trim(); });
  return out;
}

function buildDomain(rows) {
  const pick = (k) => Array.from(new Set(rows.map(r => r[k]).filter(v => v!==undefined && v!==null && v!==""))).sort((a,b)=>{
    if (typeof a === "number" && typeof b === "number") return a-b;
    return String(a).localeCompare(String(b));
  });
  return {
    k: pick("candidates"),
    choices: pick("choices"),
    level: pick("level"),
    type: pick("type"),
    partisan: pick("partisan"), // includes YES, No, DP, RP
  };
}
